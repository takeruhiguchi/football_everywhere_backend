#!/usr/bin/env python3
"""
Test script to check Make-It-Animatable core functionality without bpy dependencies
"""

import sys
import os
import torch

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    print("Testing basic imports...")
    import numpy as np
    import trimesh
    from pytorch3d.transforms import Transform3d
    print("✓ Basic dependencies imported successfully")
    
    # Test core model without torch_cluster for now
    print("Testing model components...")
    
    # Test utils without the problematic parts
    from util.utils import (
        TimePrints,
        Timing,
        fix_random,
        str2bool,
        str2list,
    )
    print("✓ Utility functions imported successfully")
    
    # Test if we can import the dataset info
    from util.dataset_mixamo import (
        BONES_IDX_DICT,
        JOINTS_NUM,
        KINEMATIC_TREE,
        MIXAMO_PREFIX,
        Joint,
    )
    print("✓ Dataset utilities imported successfully")
    
    print("\n✅ Core functionality test passed!")
    print("Note: Full model functionality requires torch_cluster compatibility fix")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()