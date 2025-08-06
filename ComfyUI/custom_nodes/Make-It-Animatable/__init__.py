#!/usr/bin/env python3
"""
Make-It-Animatable ComfyUI Integration
"""

from .comfyui_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# ComfyUI will import these automatically
WEB_DIRECTORY = "./web"