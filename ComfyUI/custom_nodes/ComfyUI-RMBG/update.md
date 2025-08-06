# ComfyUI-RMBG Update Log

## V2.1.1 (2025/03/21)
Enhanced compatibility with Transformers
  - Added support for higher versions of the transformers library (≥ 4.49.0)
  - Resolved conflicts with other models requiring higher version transformers
  - Improved error handling and more user-friendly error messages
  - If you encounter issues, you can still revert to the recommended version: `pip install transformers==4.48.3`
    
  [Transformers Version Compatibility](https://github.com/1038lab/ComfyUI-RMBG/blob/main/docs/version_compatibility.md)

## V2.1.0 (2025/03/19)
### New Features
The integration of internationalization (`i18n`) support significantly enhances ComfyUI-RMBG, enabling users worldwide to utilize background removal features in their preferred languages. This update fosters a more tailored and efficient workflow within ComfyUI-RMBG. The user interface has been improved to facilitate dynamic language switching according to user preferences. All newly introduced features are designed to be fully translatable, thereby improving accessibility for users who do not speak English.

# Supported Languages
| Language | Custom Nodes `i18n` UI |
|----------|-------|
| English  | ![node_en](https://github.com/user-attachments/assets/c20e9031-2f1a-4d7b-9037-fd30171f9d58) |
| 中文     | ![node_zh](https://github.com/user-attachments/assets/0fb77875-7693-4214-86f4-c6a92ce92497) |
| 日本語   | ![node_ja](https://github.com/user-attachments/assets/2f655484-6569-47c6-9e7c-1a175e932cc6) |
| Русский  | ![node_ru](https://github.com/user-attachments/assets/17ffbb5a-6948-460e-84cb-bca05646eb4f) |
| 한국어   | ![node_ko](https://github.com/user-attachments/assets/3899e487-6620-449b-87f4-ddb51eae9520) |
| Français | ![node_fr](https://github.com/user-attachments/assets/da45208a-2567-4905-8fe2-ee26f6431c37) |

## v2.0.0 (2025/03/13)
### New Features
- Added Load Image, Preview Image, Preview Mask, and a node that previews both the image and the mask simultaneously. This is the first phase of our toolset, with more useful tools coming in future updates.
- Reorganized the code structure for better maintainability, making it easier to navigate and update.
- Renamed certain node classes to prevent conflicts with other repositories.
- Improved category organization with a new structure: 🧪AILab/🛠️UTIL/🖼️IMAGE, making tools easier to find and use.
- Integrated predefined workflows into the ComfyUI Browse Template section, allowing users to quickly load and understand each custom node’s functionality.

![image_mask_preview](https://github.com/user-attachments/assets/5e2b2679-4b63-4db1-a6c1-3b26b6f97df3)

### Technical Improvements
- Optimized utility functions for image and mask conversion
- Improved error handling and code robustness
- Updated and changed some variable names for consistency
- Enhanced compatibility with the latest ComfyUI versions

## v1.9.3 (2025/02/24)
- Clean up the code and fix the transformers version issue `transformers>=4.35.0,<=4.48.3`

## v1.9.2 (2025/02/21)
![RMBG_V1 9 2](https://github.com/user-attachments/assets/aaf51bff-931b-47ef-b20b-0dabddc49873)
### New Features
- Added Fast Foreground Color Estimation feature
  - New `refine_foreground` option for optimizing transparent backgrounds
  - Improved edge quality and detail preservation
  - Better handling of semi-transparent regions

### Technical Improvements
- Added OpenCV dependency for advanced image processing
- Enhanced foreground refinement algorithm
- Optimized memory usage for large images
- Improved edge detection accuracy

## v1.9.1 (2025/02/20)
### Technical Updates
- Changed repository for model management to the new repository
- Reorganized models files structure for better maintainability

## v1.9.0 (2025/02/19)
![rmbg_v1 9 0](https://github.com/user-attachments/assets/a7649781-42c9-4af4-94c7-6841e9395f5a)
Add and group all BiRefNet models collections into BiRefNet node.

### New BiRefNet Models Adds
- Added `BiRefNet` general purpose model (balanced performance)
- Added `BiRefNet_512x512` model (optimized for 512x512 resolution)
- Added `BiRefNet-portrait` model (optimized for portrait/human matting)
- Added `BiRefNet-matting` model (general purpose matting)
- Added `BiRefNet-HR model` (high resolution up to 2560x2560)
- Added `BiRefNet-HR-matting` model (high resolution matting)
- Added `BiRefNet_lite` model (lightweight version for faster processing)
- Added `BiRefNet_lite-2K` model (lightweight version for 2K resolution)

### Technical Improvements
- Added FP16 (half-precision) support for better performance
- Optimized for high-resolution image processing
- Enhanced memory efficiency
- Maintained compatibility with existing workflows
- Simplified model loading through Transformers pipeline

## v1.8.0 (2025/02/07)
![BiRefNet-HR](https://github.com/user-attachments/assets/c27bf3c5-92b9-472d-b097-5fed0f182d47)
** (To ensure compatibility with the old V1.8.0 workflow, we have replaced this image with the new BiRefNet Node) (2025/03/01)

### New Model Added: BiRefNet-HR
  - Added support for BiRefNet High Resolution model
  - Trained with 2048x2048 resolution images
  - Superior performance metrics (maxFm: 0.925, MAE: 0.026)
  - Better edge detection and detail preservation
  - FP16 optimization for faster processing
  - MIT License for commercial use

![BiRefNet-HR-2](https://github.com/user-attachments/assets/12441891-0330-4972-95c2-b211fce07069)
** (To ensure compatibility with the old V1.8.0 workflow, we have replaced this image with the new BiRefNet Node) (2025/03/01)

### Technical Improvements
- Added FP16 (half-precision) support for better performance
- Optimized for high-resolution image processing
- Enhanced memory efficiency
- Maintained compatibility with existing workflows
- Simplified model loading through Transformers pipeline

### Performance Comparison
- BiRefNet-HR vs other models:
  - Higher resolution support (up to 2048x2048)
  - Better edge detection accuracy
  - Improved detail preservation
  - Optimized for high-resolution images
  - More efficient memory usage with FP16 support

## v1.7.0 (2025/02/05)
![rmbg_v1 7 0](https://github.com/user-attachments/assets/22053105-f3db-4e24-be66-ae0ad2cc248e)
### New Model Added: BEN2
- Added support for BEN2 (Background Elimination Network 2)
  - Improved performance over original BEN model
  - Better edge detection and detail preservation
  - Enhanced batch processing capabilities (up to 3 images per batch)
  - Optimized memory usage and processing speed

### Model Changes
- Updated model repository paths for BEN and BEN2
- Switched to 1038lab repositories for better maintenance and updates
- Maintained full compatibility with existing workflows

### Technical Improvements
- Implemented efficient batch processing for BEN2
- Optimized memory management for large batches
- Enhanced error handling and model loading
- Improved model switching and resource cleanup

### Comparison with Previous Models
![rmbg_v1 7 0](https://github.com/user-attachments/assets/5370305e-1b31-47ad-a1b4-852991b38f45)
- BEN2 vs BEN:
  - Better edge detection
  - Improved handling of complex backgrounds
  - More efficient batch processing
  - Enhanced detail preservation
  - Faster processing speed

## v1.6.0 (2025/01/22)

### New Face Segment Custom Node
- Added a new custom node for face parsing and segmentation
  - Support for 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
  - Precise facial feature extraction and segmentation
  - Multiple feature selection for combined segmentation
  - Same parameter controls as other RMBG nodes
  - Automatic model downloading and resource management
  - Perfect for portrait editing and facial feature manipulation

![RMBG_v1 6 0](https://github.com/user-attachments/assets/9ccefec1-4370-4708-a12d-544c90888bf2)

## v1.5.0 (2025/01/05)

### New Fashion and accessories Segment Custom Node
- Added a new custom node for fashion and accessories segmentation.
  - Capable of identifying and segmenting various fashion items such as dresses, shoes, and accessories.
  - Utilizes advanced machine learning techniques for accurate segmentation.
  - Supports real-time processing for enhanced user experience.
  - Ideal for fashion-related applications, including virtual try-ons and outfit recommendations.
  - Support for gray background color.

![RMBGv_1 5 0](https://github.com/user-attachments/assets/a250c1a6-8425-4902-b902-a6e1a8bfe959)

## v1.4.0 (2025/01/02)

### New Clothes Segment Node
- Added intelligent clothes segmentation functionality
  - Support for 18 different clothing categories (Hat, Hair, Face, Sunglasses, Upper-clothes, etc.)
  - Multiple item selection for combined segmentation
  - Same parameter controls as other RMBG nodes (process_res, mask_blur, mask_offset, background options)
  - Automatic model downloading and resource management

![rmbg_v1 4 0](https://github.com/user-attachments/assets/978c168b-03a8-4937-aa03-06385f34b820)

## v1.3.2 (2024/12/29)

### Updates
- Enhanced background handling to support RGBA output when "Alpha" is selected.
- Ensured RGB output for all other background color selections.

## v1.3.1 (2024/12/25)

### Bug Fixes
- Fixed an issue with mask processing when the model returns a list of masks.
- Improved handling of image formats to prevent processing errors.

## v1.3.0 (2024/12/23)

### New Segment (RMBG) Node
- Text-Prompted Intelligent Object Segmentation
  - Use natural language prompts (e.g., "a cat", "red car") to identify and segment target objects
  - Support for multiple object detection and segmentation
  - Perfect for precise object extraction and recognition tasks

![rmbg v1.3.0](https://github.com/user-attachments/assets/7607546e-ffcb-45e2-ab90-83267292757e)

### Supported Models
- SAM (Segment Anything Model)
  - sam_vit_h: 2.56GB - Highest accuracy
  - sam_vit_l: 1.25GB - Balanced performance
  - sam_vit_b: 375MB - Lightweight option
- GroundingDINO
  - SwinT: 694MB - Fast and efficient
  - SwinB: 938MB - Higher precision

### Key Features
- Intuitive Parameter Controls
  - Threshold: Adjust detection precision
  - Mask Blur: Smooth edges
  - Mask Offset: Expand or shrink selection
  - Background Options: Alpha/Black/White/Green/Blue/Red
- Automatic Model Management
  - Auto-download models on first use
  - Smart GPU memory handling

### Usage Examples
1. Tag-Style Prompts
   - Single object: "cat"
   - Multiple objects: "cat, dog, person"
   - With attributes: "red car, blue shirt"
   - Format: Use commas to separate multiple objects (e.g., "a, b, c")

2. Natural Language Prompts
   - Simple sentence: "a person wearing a red jacket"
   - Complex scene: "a woman in a blue dress standing next to a car"
   - With location: "a cat sitting on the sofa"
   - Format: Write a natural descriptive sentence

3. Tips for Better Results
   - For Tag Style:
     - Separate objects with commas: "chair, table, lamp"
     - Add attributes before objects: "wooden chair, glass table"
     - Keep it simple and clear
   - For Natural Language:
     - Use complete sentences
     - Include details like color, position, action
     - Be as descriptive as needed
   - Parameter Adjustments:
     - Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
     - Use mask blur for smoother edges
     - Adjust mask offset to fine-tune selection

## v1.2.2 (2024/12/12)
![RMBG1 2 2](https://github.com/user-attachments/assets/cb7b1ad0-a2ca-4369-9401-54957af6c636)

### Improvements
- Changed INSPYRENET model format from .pth to .safetensors for:
  - Better security
  - Faster loading speed (2-3x faster)
  - Improved memory efficiency
  - Better cross-platform compatibility
- Simplified node display name for better UI integration

## v1.2.1 (2024/12/02)

### New Features
- ANPG (animated PNG), AWEBP (animated WebP) and GIF supported.

https://github.com/user-attachments/assets/40ec0b27-4fa2-4c99-9aea-5afad9ca62a5

### Bug Fixes
- Fixed video processing issue

### Performance Improvements
- Enhanced batch processing in RMBG-2.0 model
- Added support for proper batch image handling
- Improved memory efficiency by optimizing image size handling

### Technical Details
- Added original size preservation for maintaining aspect ratios
- Implemented proper batch tensor processing
- Improved error handling and code robustness
- Performance gains:
  - Single image processing: ~5-10% improvement
  - Batch processing: up to 30-50% improvement (depending on batch size and GPU)

## v1.2.0 (2024/11/29)

### Major Changes
- Combined three background removal models into one unified node
- Added support for RMBG-2.0, INSPYRENET, and BEN models
- Implemented lazy loading for models (only downloads when first used)

### Model Introduction
- RMBG-2.0 ([Homepage](https://huggingface.co/briaai/RMBG-2.0))
  - Latest version of RMBG model
  - Excellent performance on complex backgrounds
  - High accuracy in preserving fine details
  - Best for general purpose background removal

- INSPYRENET ([Homepage](https://github.com/plemeri/InSPyReNet))
  - Specialized in human portrait segmentation
  - Fast processing speed
  - Good edge detection capability
  - Ideal for portrait photos and human subjects

- BEN (Background Elimination Network) ([Homepage](https://huggingface.co/PramaLLC/BEN))
  - Robust performance on various image types
  - Good balance between speed and accuracy
  - Effective on both simple and complex scenes
  - Suitable for batch processing

### Features
- Unified interface for all three models
- Common parameters for all models:
  - Sensitivity adjustment
  - Processing resolution control
  - Mask blur and offset options
  - Multiple background color options
  - Invert output option
  - Model optimization toggle

### Improvements
- Optimized memory usage with model clearing
- Enhanced error handling and user feedback
- Added detailed tooltips for all parameters
- Improved mask post-processing

### Dependencies
- Updated all package dependencies to latest stable versions
- Added support for transparent-background package
- Optimized dependency management

## v1.1.0 (2024/11/21)

### New Features
- Added background color options
  - Alpha (transparent background)
  - Black, White, Green, Blue, Red

![RMBG_v1 1 0](https://github.com/user-attachments/assets/b7cbadff-5386-4d96-bc34-a19ad34efb4b)

- Improved mask processing
  - Better detail preservation
  - Enhanced edge quality
  - More accurate segmentation
    
![rmbg version compare](https://github.com/user-attachments/assets/8339aa8e-46db-4f11-aa7b-0a710f0a1711)

- Added video batch processing
  - Support for video file background removal
  - Maintains original video framerate and resolution
  - Multiple output format support (with Alpha channel)
  - Efficient batch processing for video frames

https://github.com/user-attachments/assets/259220d3-c148-4030-93d6-c17dd5bccee1

- Added model cache management
  - Cache status checking
  - Model memory cleanup
  - Better error handling

### Parameter Updates
- Renamed 'invert_mask' to 'invert_output' for clarity
- Added sensitivity adjustment for mask strength
- Updated tooltips for better clarity

### Technical Improvements
- Optimized image processing pipeline
- Added proper model cache verification
- Improved memory management
- Better error handling and recovery
- Enhanced batch processing performance for videos

### Dependencies
- Added timm>=0.6.12,<1.0.0 for model support
- Updated requirements.txt with version constraints

### Bug Fixes
- Fixed mask detail preservation issues
- Improved mask edge quality
- Fixed memory leaks in model handling

### Usage Notes
- The 'Alpha' background option provides transparent background
- Sensitivity parameter now controls mask strength
- Model cache is checked before each operation
- Memory is automatically cleaned when switching models
- Video processing supports various formats and maintains quality
