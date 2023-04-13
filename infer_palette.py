import os
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPalettePipeline, ControlNetModel, AdapterTimePlus, UnetGenerator
from diffusers import UniPCMultistepScheduler
from torchvision import transforms
import glob
from tqdm import tqdm


def get_cond_color(cond_image, mask_size=64):
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def resize_in_buckets(pil_image):
    # find the best suited size
    buckets = [
        [512, 768],
        [768, 512],
        [512, 512],
    ]

    bucket_aspects = []
    for width, height in buckets:
        bucket_aspects.append(width / height)

    w, h = pil_image.size
    aspect = w / h
    bucket_id = np.abs([bucket_aspect - aspect for bucket_aspect in bucket_aspects]).argmin()
    return pil_image.resize(buckets[bucket_id])


def show_anns(anns, cond_image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)

    h, w = sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]

    palette = np.zeros((h, w, 3))
    mask = np.ones((h, w, 3)).astype(np.float64)
    visited = np.zeros((h, w))
    for ann in sorted_anns:
        m = ann['segmentation']
        # color_mask = np.random.random((1, 3)).tolist()[0]
        modify_m = (m * (1 - visited)) == 1
        if modify_m.sum() > 0:
            this_color = np.mean(cond_image[modify_m], 0)
            palette[modify_m] += this_color
            # for i in range(3):
            #     mask[modify_m, i] = color_mask[i]
            # assert not (mask == 1).all(), "error"
            visited[modify_m] += 1
        ann.pop('segmentation')

    palette = Image.fromarray(palette.astype(np.uint8))
    # mask = Image.fromarray((mask * 255).astype(np.uint8))
    return palette


def preprocess_sketch_and_palette(pil_image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare sketch image
    import functools
    norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
    sketch_model = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    ckpt = torch.load('/your/dir/to/anime2sketch/netG.pth')
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    sketch_model.load_state_dict(ckpt)
    sketch_model = sketch_model.to(device)

    cond_image = transform(pil_image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        sketch_cond = sketch_model(cond_image.to(device)).repeat(1, 3, 1, 1)

    # for intermediate saving
    detected_arr = sketch_cond.squeeze().permute((1, 2, 0)).cpu().numpy()
    detected_img = Image.fromarray(np.uint8(detected_arr * 255))

    if sketch_cond is None:
        sketch_cond = transform(detected_img).unsqueeze(0)

    # prepare color palette
    if mask_or_downsample == "meta_sam":
        sam_checkpoint = "/your/dir/to/sam_vit_h_4b8939.pth"
        model_type = "default"  # "vit_l"
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam, points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            points_per_batch=64)

        cond_image = np.asarray(pil_image.convert("RGB"))
        masks = mask_generator.generate(cond_image)
        palette = show_anns(masks, cond_image)

    else:
        palette = get_cond_color(pil_image, mask_size=32)

    c_palette = transform(palette.convert("RGB")).unsqueeze(0)
    return sketch_cond, c_palette, detected_img, palette


if __name__ == "__main__":
    # use cuda device
    device = "cuda:0"

    controlnet = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()
    adapter = AdapterTimePlus(cin=3 * 64, channels=[320, 640, 1280, 1280],
                              nums_rb=2, ksize=1, sk=True, use_conv=False).half()
    # adapter = None

    # choose one of them
    mask_or_downsample = "meta_sam"
    model_ckpt = "./models/color_meta_sam.pt"

    # model_ckpt = "./models/color_palette.pt"
    # mask_or_downsample = "downsample"

    model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

    # assign the weights of the controlnet and adapter separately
    controlnet_sd = {}
    adapter_sd = {}
    for k in model_sd.keys():
        if k.startswith("controlnet"):
            controlnet_sd[k.replace("controlnet.", "")] = model_sd[k]
        if k.startswith("adapter"):
            adapter_sd[k.replace("adapter.", "")] = model_sd[k]

    msg_control = controlnet.load_state_dict(controlnet_sd, strict=True)
    if adapter is not None:
        msg_adapter = adapter.load_state_dict(adapter_sd, strict=True)

    # define the inference pipline
    sdv15_path = "/your/dir/to/stable-diffusion-v1-5"
    pipe = StableDiffusionControlNetPalettePipeline.from_pretrained(
        sdv15_path,
        controlnet=controlnet,
        adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    all_files = sorted(list(glob.glob("/your/dir/to/images/*")))

    save_dir = f"./figs/output_with_palette_{mask_or_downsample}_sdv15"
    os.makedirs(save_dir, exist_ok=True)

    for fname in tqdm(all_files[:50]):
        file_name = os.path.splitext(os.path.basename(fname))[0]

        # open image
        pil_image = Image.open(fname)
        pil_image = resize_in_buckets(pil_image)

        sketch_cond, c_palette, sketch_img, palette_img = preprocess_sketch_and_palette(pil_image)

        # get text prompt
        prompt = "detailed high-quality professional image"

        # infer and save results
        generator = torch.Generator(device=device).manual_seed(2)
        output = pipe(
            prompt,
            cond_image=sketch_cond,
            adapter_image=None if adapter is None else c_palette,
            num_images_per_prompt=4,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            generator=generator,
            num_inference_steps=30,
        )

        # gather all images for storage
        imgs = output.images
        if adapter is None:
            h, w = palette_img.size
            palette_img = Image.new('RGB', (h, w))
        imgs.insert(0, palette_img)
        imgs.insert(0, sketch_img)

        grid = image_grid(imgs, 1, 4 + 2)
        grid.save(os.path.join(save_dir, f"{file_name}.png"))
