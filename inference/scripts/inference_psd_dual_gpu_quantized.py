"""Dual-GPU + Quantized inference for See-through full pipeline.

Combines dual-GPU support (from inference_utils.py) with NF4 quantization
(from inference_psd_quantized.py). All original files are untouched.

Supports NF4 (4-bit) and bf16 modes, dual-GPU UNet splitting, and
sequential pipeline lifecycle (build → run → delete → next pipeline).

Usage (from repo root):
    python inference/scripts/inference_psd_dual_gpu_quantized.py --srcp image.png --save_to_psd
    python inference/scripts/inference_psd_dual_gpu_quantized.py --quant_mode nf4 --save_to_psd
    python inference/scripts/inference_psd_dual_gpu_quantized.py --quant_mode none --no_group_offload
"""

import os.path as osp
import argparse
import sys
import os

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import json
import time
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.vae import TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, smart_resize, img_alpha_blending, validate_resolution
from utils.torch_utils import seed_everything
from utils.io_utils import json2dict, dict2json, find_all_imgs
from utils.inference_utils import further_extr

# ---------------------------------------------------------------------------
# Dual-GPU configuration
# ---------------------------------------------------------------------------
DUAL_GPU_MODE: bool = torch.cuda.device_count() >= 2

VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth',
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear',
    'tail', 'wings', 'objects'
]


def _get_device(slot: int = 0) -> str:
    if DUAL_GPU_MODE and torch.cuda.device_count() > slot:
        return f'cuda:{slot}'
    return 'cuda'


def _is_quantized(module) -> bool:
    return (getattr(module, 'is_quantized', False)
            or getattr(module, 'quantization_method', None) is not None)


def _safe_to(module, device, dtype=None):
    """Move module to device; skip dtype change for quantized modules."""
    if _is_quantized(module):
        module.to(device=device)
    elif dtype is not None:
        module.to(dtype=dtype, device=device)
    else:
        module.to(device=device)


def _print_gpu_report(label=''):
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        resv = torch.cuda.memory_reserved(i) / 1024**3
        print(f'[GPU {i}] {label}: allocated={alloc:.2f}GB  reserved={resv:.2f}GB')


# ---------------------------------------------------------------------------
# Build pipelines
# ---------------------------------------------------------------------------

def build_layerdiff_pipeline(args):
    """Build LayerDiff3D pipeline with dual-GPU + optional NF4."""
    dev0, dev1 = _get_device(0), _get_device(1)
    quant_mode = args.quant_mode
    repo = args.repo_id_layerdiff

    # --- Diagnostics ---
    n_gpu = torch.cuda.device_count()
    print(f'[GPU] device_count={n_gpu}  DUAL_GPU_MODE={DUAL_GPU_MODE}')
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f'[GPU {i}] {props.name}  free={free/1024**3:.1f}GB / {total/1024**3:.1f}GB')

    # --- Load components ---
    trans_vae = TransparentVAE.from_pretrained(repo, subfolder='trans_vae')
    unet_device_map = None

    if quant_mode == 'nf4':
        # NF4 UNet (~4GB) fits on single GPU — no device_map split needed
        print(f'[NF4] Loading pre-quantized UNet from {repo}')
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
    elif DUAL_GPU_MODE and not args.group_offload:
        # fp16 + dual-GPU: try to split UNet across GPUs
        try:
            print('[dual-GPU] Loading UNet with device_map="balanced"...')
            unet = UNetFrameConditionModel.from_pretrained(
                repo, subfolder='unet', device_map='balanced',
                torch_dtype=torch.float16,
                max_memory={0: '7000MiB', 1: '7000MiB'},
                offload_folder='workspace/offload',
            )
            unet_device_map = dict(getattr(unet, 'hf_device_map', {}))
            print(f'[dual-GPU] UNet device_map={unet_device_map}')
        except Exception as e:
            print(f'[dual-GPU] device_map failed ({e!r}), loading normally')
            unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')
    else:
        unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet')

    # --- Assemble pipeline ---
    pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
        repo, trans_vae=trans_vae, unet=unet, scheduler=None
    )

    # --- Device placement ---
    pipeline.vae.to(dtype=torch.float16, device=dev0)
    pipeline.trans_vae.to(dtype=torch.float16, device=dev0)

    # NF4 quantized text encoders can't be moved between CUDA devices
    # (CUBLAS handles are tied to the load device). Keep on dev0 for NF4;
    # cache_tag_embeds() will unload them after encoding anyway.
    if quant_mode == 'nf4':
        _safe_to(pipeline.text_encoder, dev0, torch.float16)
        _safe_to(pipeline.text_encoder_2, dev0, torch.float16)
    else:
        _safe_to(pipeline.text_encoder, dev1, torch.float16)
        _safe_to(pipeline.text_encoder_2, dev1, torch.float16)

    if unet_device_map:
        # Already dispatched via device_map
        print(f'[dual-GPU] UNet dispatched | text encoders → {dev1} | VAE → {dev0}')
    elif quant_mode == 'nf4':
        _safe_to(pipeline.unet, dev0, torch.float16)
    elif DUAL_GPU_MODE and not args.group_offload:
        # bf16 post-hoc dispatch fallback
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            pipeline.unet.to(dtype=torch.float16).cpu()
            no_split = ['BasicTransformerBlock', 'ResnetBlock2D',
                        'Transformer2DModel', 'TemporalBasicTransformerBlock']
            umap = infer_auto_device_map(
                pipeline.unet, max_memory={0: '9500MiB', 1: '7500MiB'},
                dtype=torch.float16, no_split_module_classes=no_split,
            )
            pipeline.unet = dispatch_model(pipeline.unet, device_map=umap)
            print(f'[dual-GPU] UNet post-hoc dispatched across {set(umap.values())}')
        except Exception as e:
            print(f'[dual-GPU] dispatch failed: {e!r} — single-GPU fallback')
            pipeline.unet.to(dtype=torch.float16, device=dev0)
    else:
        pipeline.unet.to(dtype=torch.float16, device=dev0)
        if DUAL_GPU_MODE:
            print(f'[dual-GPU] group_offload active — UNet on {dev0}')

    if args.group_offload:
        pipeline.enable_group_offload(dev0, num_blocks_per_group=1)

    pipeline.cache_tag_embeds()
    _print_gpu_report('after LayerDiff load')
    return pipeline


def build_marigold_pipeline(args):
    """Build Marigold depth pipeline with dual-GPU + optional NF4."""
    quant_mode = args.quant_mode
    repo = args.repo_id_depth
    marigold_dev = _get_device(1)

    if quant_mode == 'nf4':
        unet = UNetFrameConditionModel.from_pretrained(
            repo, subfolder='unet', torch_dtype=torch.float16)
        pipe = MarigoldDepthPipeline.from_pretrained(
            repo, unet=unet, torch_dtype=torch.float16)
        pipe.vae.to(device=marigold_dev)
        _safe_to(pipe.unet, marigold_dev)
        if not _is_quantized(pipe.text_encoder):
            pipe.text_encoder.to(device=marigold_dev)
    else:
        unet = UNetFrameConditionModel.from_pretrained(
            repo, subfolder='unet', torch_dtype=torch.float16)
        pipe = MarigoldDepthPipeline.from_pretrained(
            repo, unet=unet, torch_dtype=torch.float16)
        pipe.to(device=marigold_dev, dtype=torch.float16)

    # Memory-efficient attention (fp16 compatible with xformers on T4)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print('[marigold] xformers memory-efficient attention ✓')
    except Exception:
        pipe.enable_attention_slicing(slice_size=1)
        print('[marigold] xformers unavailable — attention slicing (1 head) ✓')

    if args.group_offload:
        pipe.enable_group_offload(marigold_dev, num_blocks_per_group=1)

    if DUAL_GPU_MODE:
        print(f'[dual-GPU] marigold → {marigold_dev}')

    pipe.cache_tag_embeds()
    _print_gpu_report('after Marigold load')
    return pipe


# ---------------------------------------------------------------------------
# Inference passes (identical logic to inference_utils.py v3 + v2 support)
# ---------------------------------------------------------------------------

def run_layerdiff(pipeline, imgp, save_dir, seed, num_inference_steps, resolution):
    """Run LayerDiff3D body + head passes."""
    pipeline.set_progress_bar_config(disable=False)
    saved = osp.join(save_dir, osp.splitext(osp.basename(imgp))[0])
    os.makedirs(saved, exist_ok=True)
    input_img = np.array(Image.open(imgp).convert('RGBA'))
    fullpage, pad_size, pad_pos = center_square_pad_resize(
        input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))

    rng = torch.Generator(device=_get_device(0)).manual_seed(seed)
    tag_version = pipeline.unet.get_tag_version()

    if tag_version == 'v2':
        out = pipeline(strength=1.0, num_inference_steps=num_inference_steps,
                       batch_size=1, generator=rng, guidance_scale=1.0,
                       prompt=VALID_BODY_PARTS_V2, negative_prompt='',
                       fullpage=fullpage)
        for rst, tag in zip(out.images, VALID_BODY_PARTS_V2):
            Image.fromarray(rst).save(osp.join(saved, f'{tag}.png'))

    elif tag_version == 'v3':
        _run_v3_layerdiff(pipeline, input_img, fullpage, saved, rng,
                          num_inference_steps, resolution, scale, pad_size, pad_pos)
    else:
        raise ValueError(f'Unsupported tag_version: {tag_version}')


def _run_v3_layerdiff(pipeline, input_img, fullpage, saved, rng,
                      num_inference_steps, resolution, scale, pad_size, pad_pos):
    """V3 two-pass: body then head."""

    def _crop_head(img, xywh):
        x, y, w, h = xywh
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = x, y, x + w, y + h
        if w < iw // 2:
            px = min(iw - x - w, x, w // 5)
            x1 = min(max(x - px, 0), iw)
            x2 = min(max(x + w + px, 0), iw)
        if h < ih // 2:
            py = min(ih - y - h, y, h // 5)
            y2 = min(max(y + h + py, 0), ih)
            y1 = min(max(y - py, 0), ih)
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)

    # --- Body pass ---
    body_tags = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear',
                 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
    out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                   generator=rng, guidance_scale=1.0, prompt=body_tags,
                   negative_prompt='', fullpage=fullpage, group_index=0)
    images = out.images
    for rst, tag in zip(images, body_tags):
        Image.fromarray(rst).save(osp.join(saved, f'{tag}.png'))
    head_img = images[2]

    # --- Head crop ---
    head_tags = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash',
                 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
    hx0, hy0, hw, hh = cv2.boundingRect(
        cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8)))
    hx = int(hx0 * scale) - pad_pos[0]
    hy = int(hy0 * scale) - pad_pos[1]
    hw, hh = int(hw * scale), int(hh * scale)

    input_head, (hx1, hy1, _, _) = _crop_head(input_img, [hx, hy, hw, hh])
    hx1 = int(hx1 / scale + pad_pos[0] / scale)
    hy1 = int(hy1 / scale + pad_pos[1] / scale)
    ih, iw = input_head.shape[:2]
    input_head, pad_size, pad_pos = center_square_pad_resize(
        input_head, resolution, return_pad_info=True)
    Image.fromarray(input_head).save(osp.join(saved, 'src_head.png'))

    # --- Head pass ---
    out = pipeline(strength=1.0, num_inference_steps=num_inference_steps, batch_size=1,
                   generator=rng, guidance_scale=1.0, prompt=head_tags,
                   negative_prompt='', fullpage=input_head, group_index=1)
    canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    py1, py2, px1, px2 = (np.array(
        [pad_pos[1], pad_pos[1] + ih, pad_pos[0], pad_pos[0] + iw]) / scale).astype(np.int64)
    scale_size = (int(pad_size[0] / scale), int(pad_size[1] / scale))

    for rst, tag in zip(out.images, head_tags):
        rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
        full = canvas.copy()
        full[hy1:hy1 + rst.shape[0], hx1:hx1 + rst.shape[1]] = rst
        Image.fromarray(full).save(osp.join(saved, f'{tag}.png'))


def run_marigold(pipe, srcp, save_dir, seed, resolution_depth, num_inference_steps=-1):
    """Run Marigold depth estimation."""
    pipe.set_progress_bar_config(disable=False)
    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)

    fullpage = np.array(Image.open(osp.join(saved, 'src_img.png')).convert('RGBA'))
    src_h, src_w = fullpage.shape[:2]

    if isinstance(resolution_depth, int) and resolution_depth == -1:
        resolution_depth = [src_h, src_w]
    resolution_depth = validate_resolution(resolution_depth)
    src_rescaled = resolution_depth[0] != src_h or resolution_depth[1] != src_w

    # --- Collect body-part images ---
    img_list, exist_list = [], []
    empty = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)

    compose_list = {'eyes': ['eyewhite', 'irides', 'eyelash', 'eyebrow'],
                    'hair': ['back hair', 'front hair']}
    for tag in VALID_BODY_PARTS_V2:
        tagp = osp.join(saved, f'{tag}.png')
        if osp.exists(tagp):
            exist_list.append(True)
            arr = np.array(Image.open(tagp))
            arr[..., -1][arr[..., -1] < 15] = 0
            img_list.append(arr)
        else:
            img_list.append(empty)
            exist_list.append(False)

    compose_dict = {}
    for c, clist in compose_list.items():
        imlist, taglist = [], []
        for tag in clist:
            p = osp.join(saved, tag + '.png')
            if osp.exists(p):
                arr = np.array(Image.open(p))
                arr[..., -1][arr[..., -1] < 15] = 0
                imlist.append(arr)
                taglist.append(tag)
        if imlist:
            img_list[VALID_BODY_PARTS_V2.index(c)] = img_alpha_blending(imlist, premultiplied=False)
            compose_dict[c] = {'taglist': taglist, 'imlist': imlist}

    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255
    blended_alpha = np.clip(blended_alpha, 0, 1) * 255
    blended_alpha = blended_alpha.astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)

    img_list_input = img_list
    if src_rescaled:
        img_list_input = [smart_resize(img, resolution_depth) for img in img_list]

    # --- Run depth prediction ---
    seed_everything(seed)
    kwargs = {'color_map': None, 'img_list': img_list_input}
    if num_inference_steps > 0:
        kwargs['denoising_steps'] = num_inference_steps
    pipe_out = pipe(**kwargs)
    depth_pred = pipe_out.depth_tensor.to(device='cpu', dtype=torch.float32).numpy()

    if src_rescaled:
        depth_pred = [smart_resize(d, (src_h, src_w)) for d in depth_pred]

    drawables = [{'img': im, 'depth': d} for im, d in zip(img_list, depth_pred)]
    blended = img_alpha_blending(drawables[:-1], premultiplied=False)

    # --- Save depth maps ---
    infop = osp.join(saved, 'info.json')
    info = json2dict(infop) if osp.exists(infop) else {'parts': {}}
    parts = info['parts']

    for ii, depth in enumerate(depth_pred[:-1]):
        depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        tag = VALID_BODY_PARTS_V2[ii]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for t, im in zip(compose_dict[tag]['taglist'][::-1], compose_dict[tag]['imlist'][::-1]):
                mask_local = im[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_h, src_w), 255, dtype=np.uint8)
                depth_local[mask_local] = depth_u8[mask_local]
                if np.any(mask_invis):
                    depth_local[mask_invis] = np.median(
                        depth_u8[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))])
                mask = np.bitwise_or(mask, mask_local)
                parts[t] = parts.get(t, {})
                Image.fromarray(depth_local).save(osp.join(saved, f'{t}_depth.png'))
            continue
        parts[tag] = parts.get(tag, {})
        Image.fromarray(depth_u8).save(osp.join(saved, f'{tag}_depth.png'))

    dict2json(info, infop)
    Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Dual-GPU + Quantized inference: LayerDiff body+head → Marigold depth → PSD")
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image or directory')
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--tblr_split', action='store_true',
                        help='split parts (handwear, eyes, etc) into left-right components')
    parser.add_argument('--quant_mode', type=str, default='nf4', choices=['nf4', 'none'],
                        help='quantization mode: nf4 (4-bit) or none (bf16 baseline)')
    parser.add_argument('--repo_id_layerdiff', type=str, default=None,
                        help='Override LayerDiff3D HF repo (auto-selected by quant_mode)')
    parser.add_argument('--repo_id_depth', type=str, default=None,
                        help='Override Marigold3D HF repo (auto-selected by quant_mode)')
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--num_inference_steps_depth', type=int, default=-1,
                        help='Marigold denoising steps (-1 = default)')
    parser.add_argument('--resolution_depth', type=int, default=768,
                        help='Marigold depth resolution (-1 to match layerdiff)')
    parser.add_argument('--group_offload', action='store_true', default=True,
                        help='Enable group offload (default: on)')
    parser.add_argument('--no_group_offload', action='store_false', dest='group_offload')
    parser.add_argument('--disable_progressbar', action='store_true')
    args = parser.parse_args()

    # --- Auto-select HF repos ---
    REPO_MAP = {
        'nf4': {
            'layerdiff': '24yearsold/seethroughv0.0.2_layerdiff3d_nf4',
            'depth': '24yearsold/seethroughv0.0.1_marigold_nf4',
        },
        'none': {
            'layerdiff': 'layerdifforg/seethroughv0.0.2_layerdiff3d',
            'depth': '24yearsold/seethroughv0.0.1_marigold',
        },
    }
    defaults = REPO_MAP[args.quant_mode]
    if args.repo_id_layerdiff is None:
        args.repo_id_layerdiff = defaults['layerdiff']
    if args.repo_id_depth is None:
        args.repo_id_depth = defaults['depth']

    # --- Image list ---
    srcp = args.srcp
    if osp.isdir(srcp):
        imglist = find_all_imgs(srcp, abs_path=True)
    else:
        imglist = [srcp]

    print(f"Dual-GPU + Quantized inference: quant_mode={args.quant_mode}, "
          f"dual_gpu={DUAL_GPU_MODE}, group_offload={args.group_offload}")
    print(f"  Images: {len(imglist)}, Resolution: {args.resolution}, "
          f"Steps: {args.num_inference_steps}, Seed: {args.seed}")

    torch.cuda.reset_peak_memory_stats()
    total_t0 = time.time()

    # --- Build pipelines (once, reused for all images) ---
    print('\n=== Building LayerDiff3D pipeline ===')
    seed_everything(args.seed)
    ld_pipeline = build_layerdiff_pipeline(args)

    print('\n=== Building Marigold depth pipeline ===')
    mg_pipeline = build_marigold_pipeline(args)

    # --- Process images ---
    for srcp in tqdm(imglist, desc='Processing images'):
        seed_everything(args.seed)
        srcname = osp.basename(osp.splitext(srcp)[0])
        saved = osp.join(args.save_dir, srcname)

        print(f'\n--- LayerDiff3D: {srcp} ---')
        ld_t0 = time.time()
        run_layerdiff(ld_pipeline, srcp, args.save_dir, args.seed,
                      args.num_inference_steps, args.resolution)
        print(f'  LayerDiff3D done in {time.time() - ld_t0:.1f}s')

        print(f'--- Marigold depth: {srcp} ---')
        mg_t0 = time.time()
        run_marigold(mg_pipeline, srcp, args.save_dir, args.seed,
                     resolution_depth=args.resolution_depth,
                     num_inference_steps=args.num_inference_steps_depth)
        print(f'  Marigold done in {time.time() - mg_t0:.1f}s')

        print(f'--- PSD assembly: {srcp} ---')
        psd_t0 = time.time()
        further_extr(saved, rotate=False, save_to_psd=args.save_to_psd,
                     tblr_split=args.tblr_split)
        print(f'  PSD assembly done in {time.time() - psd_t0:.1f}s')

    total_time = time.time() - total_t0

    # --- Stats ---
    stats = {
        'quant_mode': args.quant_mode,
        'dual_gpu': DUAL_GPU_MODE,
        'num_images': len(imglist),
        'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'total_time_s': total_time,
    }
    print(f'\n{"="*60}')
    print(json.dumps(stats, indent=2))
    print(f'{"="*60}')

    # Save stats next to last processed image
    stats_path = osp.join(args.save_dir, 'stats_dual_gpu_quantized.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats saved to {stats_path}')
