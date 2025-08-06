# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, and BEN. It leverages deep learning techniques
# to process images and generate masks for background removal.

# Models License Notice:
# - SAM: MIT License (https://github.com/facebookresearch/segment-anything)
# - GroundingDINO: MIT License (https://github.com/IDEA-Research/GroundingDINO)

# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import sys
import copy
import requests
from urllib.parse import urlparse

import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management
from segment_anything import sam_model_registry, SamPredictor

SAM_MODELS = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://huggingface.co/1038lab/sam/resolve/main/sam_vit_h.pth",
        "model_type": "vit_h"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/1038lab/sam/resolve/main/sam_vit_l.pth",
        "model_type": "vit_l"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://huggingface.co/1038lab/sam/resolve/main/sam_vit_b.pth",
        "model_type": "vit_b"
    }
}

DINO_MODELS = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    }
}

def normalize_array(arr):
    return arr.astype(np.float32) / 255.0

def denormalize_array(arr):
    return np.clip(255. * arr, 0, 255).astype(np.uint8)

def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0))

def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((image.height, image.width), dtype=torch.float32, device="cpu")[None,]
    return (image_rgb, mask)

def process_mask(mask_image: Image.Image, invert_output: bool = False, 
                mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)

    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    if mask_offset != 0:
        filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filter_type(size))
    
    return mask_image

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def image2mask(image: Image.Image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    return image.squeeze()

def apply_background_color(image: Image.Image, mask_image: Image.Image, 
                         background_color: str = "Alpha") -> Image.Image:
    bg_colors = {
        "Alpha": None,
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "red": (255, 0, 0)
    }

    rgba_image = image.copy().convert('RGBA')
    rgba_image.putalpha(mask_image.convert('L'))

    if background_color != "Alpha":
        bg_color = bg_colors[background_color]
        bg_image = Image.new('RGBA', image.size, (*bg_color, 255))
        composite_image = Image.alpha_composite(bg_image, rgba_image)
        return composite_image.convert('RGB')
    
    return rgba_image

class Segment:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "prompt": "Enter the object or scene you want to segment. Use tag-style or natural language for more detailed prompts.",
            "threshold": "Adjust mask detection strength (higher = more strict)",
            "mask_blur": "Apply Gaussian blur to mask edges (0 = disabled)",
            "mask_offset": "Expand/Shrink mask boundary (positive = expand, negative = shrink)",
            "background_color": "Choose background color (Alpha = transparent)",
            "invert_output": "Invert the mask output",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Object to segment", "tooltip": tooltips["prompt"]}),
                "sam_model": (list(SAM_MODELS.keys()),),
                "dino_model": (list(DINO_MODELS.keys()),),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": tooltips["threshold"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "background_color": (["Alpha", "black", "white", "gray", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background_color"]}),
                "invert_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "segment"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        from groundingdino.datasets import transforms as T
        from groundingdino.util.utils import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        
        self.T = T
        self.clean_state_dict = clean_state_dict
        self.SLConfig = SLConfig
        self.build_model = build_model

    def segment(self, image, prompt, sam_model, dino_model, threshold=0.35,
                mask_blur=0, mask_offset=0, background_color="Alpha", 
                invert_output=False):
        print(f'Processing create segment for: "{prompt}"...')
        
        image = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
        dino_model = self.load_groundingdino(dino_model)
        sam_model = self.load_sam(sam_model)
        boxes = self.predict_boxes(dino_model, image, prompt, threshold)
        
        if boxes is None or boxes.shape[0] == 0:
            print(f'No objects found for: "{prompt}"')
            width, height = image.size
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        
        masks = self.generate_masks(sam_model, image, boxes)
        if masks is None:
            print(f'Failed to generate mask for: "{prompt}"')
            width, height = image.size
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)

        mask_image = Image.fromarray((masks[1][0].numpy() * 255).astype(np.uint8))
        mask_image = process_mask(mask_image, invert_output, mask_blur, mask_offset)
        
        result_image = apply_background_color(image, mask_image, background_color)
        
        if background_color != "Alpha":
            result_image = result_image.convert("RGB")
        else:
            result_image = result_image.convert("RGBA")

        print(f'Successfully created segment for: "{prompt}"')
        return (pil2tensor(result_image), image2mask(mask_image))

    def load_sam(self, model_name):
        sam_checkpoint_path = self.get_local_filepath(
            SAM_MODELS[model_name]["model_url"], "sam")
        model_type = SAM_MODELS[model_name]["model_type"]
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam_device = comfy.model_management.get_torch_device()
        sam.to(device=sam_device)
        sam.eval()
        return sam

    def load_groundingdino(self, model_name):
        import sys
        from io import StringIO
        temp_stdout = StringIO()
        original_stdout = sys.stdout
        sys.stdout = temp_stdout

        try:
            dino_model_args = self.SLConfig.fromfile(
                self.get_local_filepath(
                    DINO_MODELS[model_name]["config_url"],
                    "grounding-dino"
                )
            )
            dino = self.build_model(dino_model_args)
            checkpoint = torch.load(
                self.get_local_filepath(
                    DINO_MODELS[model_name]["model_url"],
                    "grounding-dino"
                )
            )
            dino.load_state_dict(self.clean_state_dict(checkpoint['model']), strict=False)
            device = comfy.model_management.get_torch_device()
            dino.to(device=device)
            dino.eval()
            return dino
        finally:
            output = temp_stdout.getvalue()
            sys.stdout = original_stdout
            
            for line in output.split('\n'):
                if 'error' in line.lower():
                    print(line)

    def _load_dino_image(self, image_pil):
        transform = self.T.Compose([
            self.T.RandomResize([800], max_size=1333),
            self.T.ToTensor(),
            self.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image

    def _get_grounding_output(self, model, image, caption, box_threshold):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        return boxes_filt.cpu()

    def predict_boxes(self, model, image, prompt, threshold):
        dino_image = self._load_dino_image(image.convert("RGB"))
        boxes_filt = self._get_grounding_output(model, dino_image, prompt, threshold)
        H, W = image.size[1], image.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        return boxes_filt

    def generate_masks(self, model, image, boxes):
        if boxes.shape[0] == 0:
            return None
            
        if not hasattr(self, 'predictor'):
            self.predictor = SamPredictor(model)
            
        image_np = np.array(image)
        image_np_rgb = image_np[..., :3]
        
        self.predictor.set_image(image_np_rgb)
        
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(comfy.model_management.get_torch_device()),
            multimask_output=False
        )
        
        return create_tensor_output(image_np, masks.permute(1, 0, 2, 3).cpu().numpy(), boxes)


    def get_local_filepath(self, url, dirname, local_file_name=None):
        if not local_file_name:
            local_file_name = os.path.basename(urlparse(url).path)

        destination = folder_paths.get_full_path(dirname, local_file_name)
        if destination:
            return destination

        folder = os.path.join(folder_paths.models_dir, dirname)
        os.makedirs(folder, exist_ok=True)

        destination = os.path.join(folder, local_file_name)
        if not os.path.exists(destination):
            try:
                download_url_to_file(url, destination)
            except Exception as e:
                if os.path.exists(destination):
                    os.remove(destination)
                raise Exception(f'Failed to download model from {url}: {str(e)}')
        return destination

NODE_CLASS_MAPPINGS = {
    "Segment": Segment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Segment": "Segment (RMBG)"
}