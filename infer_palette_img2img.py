import os
import torch
from PIL import Image
import numpy as np
from diffusers import ControlNetModel, LineartDetector, StableDiffusionImg2ImgControlNetPalettePipeline
from diffusers import UniPCMultistepScheduler
from torchvision import transforms
import glob
from tqdm import tqdm
import einops
from infer_palette import get_cond_color, show_anns, image_grid, HWC3, resize_in_buckets, SAMImageAnnotator


def preprocess_sketch_and_palette(pil_image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    sketch_model = LineartDetector().to(device)
    cond_image = np.asarray(pil_image.convert("RGB"))[None, ...]
    cond_image = torch.from_numpy(cond_image / 255.0).float()
    cond_image = einops.rearrange(cond_image, "b  h w c -> b c h w").clone()

    with torch.no_grad():
        sketch_cond = sketch_model.model_coarse(cond_image.to(device)).repeat(1, 3, 1, 1)

    sketch_cond = torch.clip(sketch_cond, min=0.0, max=1.0)
    sketch_cond = 1.0 - sketch_cond

    # for intermediate saving
    detected_arr = sketch_cond.squeeze().permute((1, 2, 0)).cpu().numpy()  # [-1, 1]
    detected_arr = (detected_arr * 255.0).clip(0, 255.0)
    detected_img = Image.fromarray(np.uint8(detected_arr))

    # prepare color palette
    sam_palette = sam_annotator(pil_image) if isinstance(sam_annotator, SAMImageAnnotator) else None
    rect_palette = get_cond_color(pil_image, mask_size=32)

    c_palette = transform(rect_palette.convert("RGB")).unsqueeze(0)
    return sketch_cond, c_palette, detected_img, rect_palette, sam_palette


if __name__ == "__main__":
    # use cuda device
    device = "cuda:0"
    controlnet = ControlNetModel.from_config("./model_configs/controlnet_config.json").half()
    adapter = ControlNetModel.from_pretrained("./model_configs/controlnet_config.json").half()

    sketch_method = "skmodel"
    sam_annotator = SAMImageAnnotator()

    model_ckpt = f"./models/color_img2img_palette.pt"
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
    print(f"msg_control: {msg_control} ")
    if adapter is not None:
        msg_adapter = adapter.load_state_dict(adapter_sd, strict=False)
        print(f"msg_adapter: {msg_adapter} ")

    # define the inference pipline
    sdv15_path = "/your/dir/to/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgControlNetPalettePipeline.from_pretrained(
        sdv15_path,
        controlnet=controlnet,
        adapter=adapter,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    all_files = sorted(list(glob.glob("/your/dir/to/images/*")))

    save_dir = "./figs"
    os.makedirs(save_dir, exist_ok=True)

    for fname in tqdm(all_files[:50]):
        file_name = os.path.splitext(os.path.basename(fname))[0]

        # open image
        pil_image = Image.open(fname)
        pil_image = resize_in_buckets(pil_image)
        sketch_cond, c_palette, sketch_img, palette_img, palette_sam = preprocess_sketch_and_palette(pil_image)

        # get text prompt
        prompt = "detailed high-quality professional image"

        # infer and save results
        generator = torch.Generator(device=device).manual_seed(2)
        output = pipe(
            prompt=prompt,
            image=palette_sam,
            strength=0.75,  # this param might need tweak
            cond_image=sketch_cond,
            adapter_image=c_palette,
            use_controlnet_as_adapter=isinstance(adapter, ControlNetModel),
            num_images_per_prompt=4,
            negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
            generator=generator,
            num_inference_steps=30,
        )

        # gather all images for storage
        imgs = output.images
        if adapter is None:
            h, w = palette_img.size
            palette_img = Image.new('RGB', (h, w))
        if palette_sam is None:
            h, w = palette_img.size
            palette_sam = Image.new('RGB', (h, w))
        imgs.insert(0, palette_sam)
        imgs.insert(0, palette_img)
        imgs.insert(0, sketch_img)

        grid = image_grid(imgs, 1, len(imgs))
        grid.save(os.path.join(save_dir, f"{file_name}.png"))
