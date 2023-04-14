import functools
from PIL import Image
import numpy as np
import gradio as gr
import torch
from torchvision import transforms

from diffusers import StableDiffusionControlNetPalettePipeline, ControlNetModel, UniPCMultistepScheduler, AdapterTimePlus, UnetGenerator
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from infer_palette import resize_in_buckets, show_anns


device = "cuda:0"
MODEL_TYPE = "block" # one of ['sam', 'block']
assert MODEL_TYPE in ['sam', 'block']

SD15_CKPT_PATH = "/your/dir/to/stable-diffusion-v1-5" # in diffuser format
BLOCK_PALETTE_MODEL_CKPT_PATH = "/your/dir/to/color_palette.pt"
SAM_PALETTE_MODEL_CKPT_PATH = "/your/dir/to/color_meta_sam.pt" 
CONTROLNET_INIT_CKPT_PATH = "/your/dir/to/sd15-controlnet-init"  # in diffuser format
ANIME2SKETCH_CKPT_PATH = "/your/dir/to/ANIME2Sketch/netG.pth"
SAM_CKPT = "/your/dir/to/sam_vit_h_4b8939.pth"

model_type = "default"
sam = sam_model_registry[model_type](checkpoint=SAM_CKPT)
sam.to(device=device)

if MODEL_TYPE == 'block':
    MODEL_CKPT_PATH = BLOCK_PALETTE_MODEL_CKPT_PATH
elif MODEL_TYPE == 'sam':
    MODEL_CKPT_PATH = SAM_PALETTE_MODEL_CKPT_PATH
else:
    ValueError

class ColorizeModel:
    def __init__(self, device) -> None:
        self.device = device
                
        self.controlnet = ControlNetModel.from_pretrained(CONTROLNET_INIT_CKPT_PATH, torch_dtype=torch.float16)
        self.adapter = AdapterTimePlus(cin=3 * 64, channels=[320, 640, 1280, 1280],
                              nums_rb=2, ksize=1, sk=True, use_conv=False).half()

        model_ckpt = torch.load(MODEL_CKPT_PATH, map_location="cpu")["module"]
        controlnet_sd = {}
        adapter_sd = {}
        for k in model_ckpt.keys():
            if k.startswith("controlnet"):
                controlnet_sd[k.replace("controlnet.", "")] = model_ckpt[k]
            if k.startswith("adapter"):
                adapter_sd[k.replace("adapter.", "")] = model_ckpt[k]
        
        msg_control = self.controlnet.load_state_dict(controlnet_sd, strict=True)      
        msg_adapter = self.adapter.load_state_dict(adapter_sd, strict=True)
        print(f"msg_control: {msg_control} \n"
                f"msg_adapter: {msg_adapter}")        
             
        self.pipe = StableDiffusionControlNetPalettePipeline.from_pretrained(SD15_CKPT_PATH,controlnet=self.controlnet, adapter=self.adapter,torch_dtype=torch.float16, safety_checker=None,).to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.init_sketch_model()


    def init_sketch_model(self):
        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)
        anime2Sketch = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
        anime2Sketch_ckpt = torch.load(ANIME2SKETCH_CKPT_PATH)
        for key in list(anime2Sketch_ckpt.keys()):
            if 'module.' in key:
                anime2Sketch_ckpt[key.replace('module.', '')] = anime2Sketch_ckpt[key]
                del anime2Sketch_ckpt[key]
        anime2Sketch.load_state_dict(anime2Sketch_ckpt)

        self.sketch_model = anime2Sketch.to(self.device)


    @torch.inference_mode()
    def colorize(self, sketch_source, sketch, palette, prompt, negative_prompt, ip_cfg_scale, ip_seed, ip_steps, ip_num_images):
        if isinstance(self.sketch_model, UnetGenerator):
            sketch_source = resize_in_buckets(sketch_source)
            sketch_source = self.transform(sketch_source.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                sketch = self.sketch_model(sketch_source.to(self.device)).repeat(1, 3, 1, 1) # input:[-1,1] output:[-1,1]
        else:
            NotImplementedError

        ip_height, ip_width = sketch.shape[2], sketch.shape[3] 

        if palette is not None:
            palette = palette.convert("RGB").resize((ip_width, ip_height))
            palette = self.transform(palette).unsqueeze(0)

        generator = torch.Generator(device=self.device).manual_seed(int(ip_seed)) if int(ip_seed) >= 0 else None
        output = self.pipe(
            prompt,
            cond_image=sketch,
            adapter_image=palette,
            num_images_per_prompt=ip_num_images,
            negative_prompt=negative_prompt+"monochrome, lowres, bad anatomy, worst quality, low quality",
            generator=generator,
            num_inference_steps=ip_steps,
            width=ip_width, 
            height=ip_height,
            guidance_scale=ip_cfg_scale,
        ).images
        
        output = [np.array(image, dtype=np.uint8) for image in output]

        return output
    

    @torch.inference_mode()
    def get_cond_anime2sketch(self, pil_image):
        pil_image = resize_in_buckets(pil_image.copy())
        cond_image = self.transform(pil_image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            sketch_cond = self.sketch_model(cond_image.to(self.device)).repeat(1, 3, 1, 1) # input:[-1,1] output:[-1,1]

        # for intermediate saving
        detected_arr = sketch_cond.squeeze().permute((1, 2, 0)).cpu().numpy()# [-1, 1]
        detected_arr = (detected_arr + 1) / 2.0 * 255.0
        detected_img = Image.fromarray(np.uint8(detected_arr))

        return detected_img


    def get_cond_palette(self, type, cond_image, block_size=64):
        cond_image = resize_in_buckets(cond_image)
        if type == "block":
            return self.get_block_palette(cond_image, block_size)
        elif type == "sam":
            return self.get_sam_color(cond_image)
        else:
            NotImplementedError
        
        
    def get_block_palette(self, cond_image, block_size=64):
        H, W = cond_image.size
        cond_image = cond_image.resize((W // block_size, H // block_size), Image.BICUBIC)
        color = cond_image.resize((H, W), Image.NEAREST)

        return color
    

    @torch.inference_mode()
    def get_sam_color(self, cond_image):
        mask_generator = SamAutomaticMaskGenerator(
            model=sam, points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            points_per_batch=64)

        cond_image = np.asarray(cond_image.convert("RGB"))
        masks = mask_generator.generate(cond_image)
        palette = show_anns(masks, cond_image)

        return palette


colorize_model = ColorizeModel(f"{device}") 


def palette_generate(palette_source, palette_processor):
    if palette_processor == "Color":
        palette_source = resize_in_buckets(palette_source)
        return palette_source
    elif palette_processor == "Block palette":
        return colorize_model.get_cond_palette(type="block", cond_image=palette_source, block_size=32)    
    elif palette_processor == "SAM palette":
        return colorize_model.get_cond_palette(type="sam", cond_image=palette_source)  
    if palette_processor == "Nothing":
        pass
    else:
        NotImplementedError

def sketch_generate(sketch_source, sketch_processor):
    if sketch_processor == "Nothing":
        return sketch_source
    elif sketch_processor == "Anime2sketch from Image":
        return colorize_model.get_cond_anime2sketch(sketch_source)
    else:
        NotImplementedError


def colorize_generate(sketch_source, sketch, palette, prompt, negative_prompt, ip_cfg_scale, ip_seed, ip_steps, ip_num_images):
    output = colorize_model.colorize(sketch_source, sketch, palette, prompt, negative_prompt, ip_cfg_scale, ip_seed, ip_steps, ip_num_images)
    return output


block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column(min_width=50):
                    sketch_source = gr.Image(source='upload', type="pil", image_mode="RGB", label="sketch source")
                    with gr.Row():
                        sketch_processor = gr.Dropdown(choices=["Anime2sketch from Image", "Nothing"], value="Anime2sketch from Image", label="Input type for structure")

                with gr.Column(min_width=50):
                    palette_source = gr.Image(source='upload', type="pil", image_mode="RGB", label="palette source")  
                    if MODEL_TYPE == "block":
                        palette_processor = gr.Dropdown(choices=["Block palette", "Nothing"], value="Block palette", label="Input type for color")                        
                    elif MODEL_TYPE == "sam":
                        palette_processor = gr.Dropdown(choices=["SAM palette", "Nothing"], value="Block palette", label="Input type for color")
            
        with gr.Column(min_width=300):
            prompt = gr.Textbox(label="Prompt", elem_id=f"colorization_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")
            negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"colorization_neg_prompt", show_label=False, lines=2, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")
            with gr.Row():
                with gr.Column(min_width=50):
                    ip_steps = gr.Slider(minimum=1, maximum=150, step=1, label="sampling steps", value=50)
                with gr.Column(min_width=50):
                    ip_cfg_scale = gr.Slider(minimum=2.0, maximum=15., step=0.5, label='CFG Scale', value=7.5)
            with gr.Row():
                with gr.Column(min_width=50):
                    ip_seed = gr.Number(label='Seed', value=-1)
                with gr.Column(min_width=50):
                    ip_num_images = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                get_sketch_button = gr.Button("Generate Sketch")
                get_palette_button = gr.Button("Generate Palette")

        with gr.Column():
            colorize_button = gr.Button("Colorize")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                sketch = gr.Image(source='upload', type="pil", image_mode="RGB", label="input_sketch", interactive=True)
                palette = gr.Image(source='upload', type="pil", image_mode="RGB", label="input_palette")
        
        with gr.Column():
            ip_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            
    get_sketch_button.click(sketch_generate, inputs=[sketch_source, sketch_processor], outputs=[sketch])
    get_palette_button.click(palette_generate, inputs=[palette_source, palette_processor], outputs=[palette])
    colorize_button.click(colorize_generate,
        inputs=[sketch_source, sketch, palette, prompt, negative_prompt, ip_cfg_scale, ip_seed, ip_steps, ip_num_images],
        outputs=[ip_gallery])


block.launch(server_name='0.0.0.0', server_port=8081)
