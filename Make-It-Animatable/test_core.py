#!/usr/bin/env python3
"""
Test core functionality without blender dependencies
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("Testing basic imports...")
    import trimesh
    from pytorch3d.transforms import Transform3d
    print("✓ Trimesh and PyTorch3D imported successfully")
    
    # Test basic utils that don't depend on blender
    from util.utils import (
        TimePrints,
        Timing,
        fix_random,
        str2bool,
        str2list,
    )
    print("✓ Basic utility functions imported successfully")
    
    # Test if we can create some core structures without full model
    print("Testing core structures...")
    
    # Test basic tensor operations that the model would use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor = torch.randn(10, 3).to(device)
    print(f"✓ Created test tensor on {device}: {test_tensor.shape}")
    
    # Test mesh loading with trimesh
    print("Testing mesh operations...")
    # Create a simple test mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"✓ Created test mesh with {len(mesh.vertices)} vertices")
    
    print("\n✅ Core functionality test passed!")
    print("Ready to create ComfyUI wrapper for animation processing")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()