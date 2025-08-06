# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, and BEN. It leverages deep learning techniques
# to process images and generate masks for background removal.
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG


import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageFilter
import onnxruntime
import folder_paths
from huggingface_hub import hf_hub_download
import shutil
from torchvision import transforms

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def image2mask(image: Image.Image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = pil2tensor(image)
    return image.squeeze()[..., 0]

def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

class BodySegment:
    def __init__(self):
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "body_segment")
        self.model_file = "deeplabv3p-resnet50-human.onnx"
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = [
            "Hair", "Glasses", "Top-clothes", "Bottom-clothes", 
            "Torso-skin", "Face", "Left-arm", "Right-arm",
            "Left-leg", "Right-leg", "Left-foot", "Right-foot"
        ]
        
        tooltips = {
            "process_res": "Processing resolution (fixed at 512x512)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "background_color": "Choose background color (Alpha = transparent)",
            "invert_output": "Invert both image and mask output",
        }
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False}) 
                   for cls_name in available_classes},
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "background_color": (["Alpha", "black", "white", "gray", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background_color"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "segment_body"
    CATEGORY = "🧪AILab/🧽RMBG"

    def check_model_cache(self):
        model_path = os.path.join(self.cache_dir, self.model_file)
        if not os.path.exists(model_path):
            return False, "Model file not found"
        return True, "Model cache verified"

    def clear_model(self):
        if self.model is not None:
            del self.model
            self.model = None

    def download_model_files(self):
        model_id = "Metal3d/deeplabv3p-resnet50-human"
        os.makedirs(self.cache_dir, exist_ok=True)
        print("Downloading body segmentation model...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=self.model_file,
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            if os.path.dirname(downloaded_path) != self.cache_dir:
                target_path = os.path.join(self.cache_dir, self.model_file)
                shutil.move(downloaded_path, target_path)
            return True, "Model file downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model file: {str(e)}"

    def segment_body(self, images, mask_blur=0, mask_offset=0, background_color="Alpha", invert_output=False, **class_selections):
        try:
            # Check and download model if needed
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"Cache check: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)
            
            # Load model if needed
            if self.model is None:
                self.model = onnxruntime.InferenceSession(
                    os.path.join(self.cache_dir, self.model_file)
                )

            # Class mapping
            class_map = {
                "Hair": 2, "Glasses": 4, "Top-clothes": 5,
                "Bottom-clothes": 9, "Torso-skin": 10, "Face": 13,
                "Left-arm": 14, "Right-arm": 15, "Left-leg": 16,
                "Right-leg": 17, "Left-foot": 18, "Right-foot": 19
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Face", "Hair", "Top-clothes", "Bottom-clothes"]

            batch_tensor = []
            batch_masks = []
            
            for image in images:
                orig_image = tensor2pil(image)
                w, h = orig_image.size
                
                # Resize to 512x512 (model requirement)
                input_image = orig_image.resize((512, 512))
                input_array = np.array(input_image).astype(np.float32) / 127.5 - 1

                # Add batch dimension
                input_array = np.expand_dims(input_array, axis=0)

                # Run inference
                input_name = self.model.get_inputs()[0].name
                output_name = self.model.get_outputs()[0].name
                result = self.model.run([output_name], {input_name: input_array})

                # Process results
                result = np.array(result[0])
                pred_seg = result.argmax(axis=3).squeeze(0)

                # Combine selected class masks
                combined_mask = np.zeros_like(pred_seg, dtype=np.float32)
                for class_name in selected_classes:
                    mask = (pred_seg == class_map[class_name]).astype(np.float32)
                    combined_mask = np.clip(combined_mask + mask, 0, 1)

                # Convert to PIL and resize back to original size
                mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))
                mask_image = mask_image.resize((w, h), Image.Resampling.LANCZOS)

                if mask_blur > 0:
                    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                if mask_offset != 0:
                    if mask_offset > 0:
                        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=mask_offset * 2 + 1))
                    else:
                        mask_image = mask_image.filter(ImageFilter.MinFilter(size=-mask_offset * 2 + 1))

                if invert_output:
                    mask_image = Image.fromarray(255 - np.array(mask_image))

                # Handle background color
                if background_color == "Alpha":
                    rgba_image = RGB2RGBA(orig_image, mask_image)
                    result_image = pil2tensor(rgba_image)
                else:
                    bg_colors = {
                        "black": (0, 0, 0),
                        "white": (255, 255, 255),
                        "gray": (128, 128, 128),
                        "green": (0, 255, 0),
                        "blue": (0, 0, 255),
                        "red": (255, 0, 0)
                    }
                    
                    rgba_image = RGB2RGBA(orig_image, mask_image)
                    bg_image = Image.new('RGBA', orig_image.size, (*bg_colors[background_color], 255))
                    composite_image = Image.alpha_composite(bg_image, rgba_image)
                    result_image = pil2tensor(composite_image.convert('RGB'))

                batch_tensor.append(result_image)
                batch_masks.append(pil2tensor(mask_image))

            # Prepare final output
            batch_tensor = torch.cat(batch_tensor, dim=0)
            batch_masks = torch.cat(batch_masks, dim=0)
            
            return (batch_tensor, batch_masks)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in Body Segmentation processing: {str(e)}")
        finally:
            self.clear_model()

NODE_CLASS_MAPPINGS = {
    "BodySegment": BodySegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BodySegment": "Body Segment (RMBG)"
} 