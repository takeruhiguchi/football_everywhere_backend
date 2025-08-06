#!/usr/bin/env python3
"""
FastAPI server for Make-It-Animatable
Runs in isolated environment to avoid library conflicts with ComfyUI
"""

import os
import sys
import tempfile
import uuid
import json
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import uvicorn
import torch
import numpy as np
import trimesh
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pytorch3d.transforms import Transform3d

# Add Make-It-Animatable to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Lazy imports to avoid bpy conflicts during startup
# These will be imported when needed

def lazy_import_make_it_animatable():
    """Lazy import Make-It-Animatable modules to avoid startup conflicts"""
    global PCAE, BONES_IDX_DICT, JOINTS_NUM, KINEMATIC_TREE, MIXAMO_PREFIX
    global TEMPLATE_PATH, Joint, get_hips_transform, BONES_IDX_DICT_ADD
    global JOINTS_NUM_ADD, KINEMATIC_TREE_ADD, TEMPLATE_PATH_ADD
    global TimePrints, Timing, apply_transform, fix_random, get_normalize_transform
    global load_gs, make_archive, pose_local_to_global, pose_rot_to_global
    global sample_mesh, save_gs, str2bool, str2list, to_pose_local, to_pose_matrix, transform_gs
    
    try:
        from model import PCAE
        from util.dataset_mixamo import (
            BONES_IDX_DICT,
            JOINTS_NUM,
            KINEMATIC_TREE,
            MIXAMO_PREFIX,
            TEMPLATE_PATH,
            Joint,
            get_hips_transform,
        )
        from util.dataset_mixamo_additional import BONES_IDX_DICT as BONES_IDX_DICT_ADD
        from util.dataset_mixamo_additional import JOINTS_NUM as JOINTS_NUM_ADD
        from util.dataset_mixamo_additional import KINEMATIC_TREE as KINEMATIC_TREE_ADD
        from util.dataset_mixamo_additional import TEMPLATE_PATH as TEMPLATE_PATH_ADD
        from util.utils import (
            TimePrints,
            Timing,
            apply_transform,
            fix_random,
            get_normalize_transform,
            load_gs,
            make_archive,
            pose_local_to_global,
            pose_rot_to_global,
            sample_mesh,
            save_gs,
            str2bool,
            str2list,
            to_pose_local,
            to_pose_matrix,
            transform_gs,
        )
        return True
    except Exception as e:
        print(f"❌ Failed to import Make-It-Animatable modules: {e}")
        return False

# Global placeholders
PCAE = None
BONES_IDX_DICT = None
JOINTS_NUM = None
KINEMATIC_TREE = None
MIXAMO_PREFIX = None
TEMPLATE_PATH = None
Joint = None
get_hips_transform = None
BONES_IDX_DICT_ADD = None
JOINTS_NUM_ADD = None
KINEMATIC_TREE_ADD = None
TEMPLATE_PATH_ADD = None
TimePrints = None
Timing = None
apply_transform = None
fix_random = None
get_normalize_transform = None
load_gs = None
make_archive = None
pose_local_to_global = None
pose_rot_to_global = None
sample_mesh = None
save_gs = None
str2bool = None
str2list = None
to_pose_local = None
to_pose_matrix = None
transform_gs = None

# FastAPI app will be defined after lifespan handler

# Global variables for models and configuration
models_initialized = False
device = None
N = 32768
hands_resample_ratio = 0.5
geo_resample_ratio = 0.0
bw_additional = False
joints_additional = False
bones_idx_dict_bw = None
bones_idx_dict_joints = None
model_bw = None
model_bw_normal = None
model_joints = None
model_joints_add = None
model_coarse = None
model_pose = None

@dataclass()
class DB:
    """Database-like class to store processing state (from app.py)"""
    mesh: trimesh.Trimesh = None
    gs: torch.Tensor = None
    gs_rest: torch.Tensor = None
    is_mesh: bool = None
    sample_mask: np.ndarray = None
    verts: torch.Tensor = None
    verts_normal: torch.Tensor = None
    faces: np.ndarray = None
    pts: torch.Tensor = None
    pts_normal: torch.Tensor = None
    global_transform: Transform3d = None

    output_dir: str = None
    joints_coarse_path: str = None
    normed_path: str = None
    sample_path: str = None
    bw_path: str = None
    joints_path: str = None
    rest_lbs_path: str = None
    rest_vis_path: str = None
    anim_path: str = None
    anim_vis_path: str = None

    bw: torch.Tensor = None
    joints: torch.Tensor = None
    joints_tail: torch.Tensor = None
    pose: torch.Tensor = None

    def clear(self):
        for k in self.__dict__:
            self.__dict__[k] = None
        return self

class AnimationRequest(BaseModel):
    """Request model for animation parameters"""
    # Input settings
    is_gs: bool = False
    opacity_threshold: float = 0.01
    no_fingers: bool = True
    rest_pose_type: str = "No"  # "T-pose", "A-pose", "大-pose", "No"
    rest_parts: List[str] = []  # ["Fingers", "Arms", "Legs", "Head"]
    
    # Weight settings
    input_normal: bool = False
    bw_fix: bool = True
    bw_vis_bone: str = "LeftArm"
    
    # Animation settings
    reset_to_rest: bool = True
    retarget: bool = True
    inplace: bool = True

def clear_memory(db: DB = None):
    """Clear memory and GPU cache"""
    if db is not None:
        db.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared")

def init_models():
    """Initialize all models (adapted from app.py)"""
    global device, N, hands_resample_ratio, geo_resample_ratio, bw_additional, joints_additional
    global bones_idx_dict_bw, bones_idx_dict_joints, model_bw, model_bw_normal, model_joints
    global model_joints_add, model_coarse, model_pose, models_initialized
    
    try:
        print("🔧 Initializing Make-It-Animatable models...")
        
        # First, do lazy import
        if not lazy_import_make_it_animatable():
            raise RuntimeError("Failed to import Make-It-Animatable modules")
        
        # Change to parent directory (Make-It-Animatable root)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        print(f"Changing to parent directory: {parent_dir}")
        os.chdir(parent_dir)
        
        fix_random()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        IS_HF_ZEROGPU = str2bool(os.getenv("SPACES_ZERO_GPU", False))
        
        N = 32768
        hands_resample_ratio = 0.5
        geo_resample_ratio = 0.0
        hierarchical_ratio = hands_resample_ratio + geo_resample_ratio
        
        ADDITIONAL_BONES = bw_additional = joints_additional = False
        
        # Initialize blend weights model
        print("📦 Loading blend weights model...")
        model_bw = PCAE(
            N=N,
            input_normal=False,
            deterministic=True,
            hierarchical_ratio=hierarchical_ratio,
            output_dim=JOINTS_NUM_ADD if ADDITIONAL_BONES else JOINTS_NUM,
        )
        if ADDITIONAL_BONES:
            model_bw.load("output/vroid/bw.pth")
        else:
            model_bw.load("output/best/new/bw.pth")
        model_bw.to("cpu" if IS_HF_ZEROGPU else device).eval()
        
        # Initialize blend weights model with normals
        print("📦 Loading blend weights model (with normals)...")
        model_bw_normal = PCAE(
            N=N,
            input_normal=True,
            input_attention=True,
            deterministic=True,
            hierarchical_ratio=hierarchical_ratio,
            output_dim=JOINTS_NUM_ADD if ADDITIONAL_BONES else JOINTS_NUM,
        )
        if ADDITIONAL_BONES:
            model_bw_normal.load("output/vroid/bw_normal.pth")
        else:
            model_bw_normal.load("output/best/new/bw_normal.pth")
        bones_idx_dict_bw = BONES_IDX_DICT_ADD if bw_additional else BONES_IDX_DICT
        model_bw_normal.to("cpu" if IS_HF_ZEROGPU else device).eval()
        
        # Initialize joints model
        print("📦 Loading joints model...")
        model_joints = PCAE(
            N=N,
            input_normal=False,
            deterministic=True,
            hierarchical_ratio=hierarchical_ratio,
            output_dim=JOINTS_NUM,
            kinematic_tree=KINEMATIC_TREE,
            predict_bw=False,
            predict_joints=True,
            predict_joints_tail=True,
            joints_attn_causal=True,
        )
        model_joints.load("output/best/new/joints.pth")
        model_joints.to(device).eval()
        bones_idx_dict_joints = BONES_IDX_DICT_ADD if joints_additional else BONES_IDX_DICT
        
        # Additional joints model if needed
        if ADDITIONAL_BONES:
            print("📦 Loading additional joints model...")
            model_joints_add = PCAE(
                N=N,
                input_normal=False,
                deterministic=True,
                hierarchical_ratio=hierarchical_ratio,
                output_dim=JOINTS_NUM_ADD,
                kinematic_tree=KINEMATIC_TREE_ADD,
                predict_bw=False,
                predict_joints=True,
                predict_joints_tail=True,
            )
            model_joints_add.load("output/vroid/joints.pth")
            model_joints_add.to(device).eval()
        
        # Initialize coarse joints model
        print("📦 Loading coarse joints model...")
        model_coarse = PCAE(
            N=N,
            input_normal=False,
            deterministic=True,
            output_dim=JOINTS_NUM,
            predict_bw=False,
            predict_joints=True,
            predict_joints_tail=True,
        )
        model_coarse.load("output/best/new/joints_coarse.pth")
        model_coarse.to(device).eval()
        
        # Initialize pose model
        print("📦 Loading pose model...")
        model_pose = PCAE(
            N=N,
            input_normal=False,
            deterministic=True,
            hierarchical_ratio=hierarchical_ratio,
            output_dim=JOINTS_NUM,
            kinematic_tree=KINEMATIC_TREE,
            predict_bw=False,
            predict_pose_trans=True,
            pose_mode="ortho6d",
            pose_input_joints=True,
            pose_attn_causal=True,
        )
        model_pose.load("output/best/new/pose.pth")
        model_pose.to(device).eval()
        
        clear_memory()
        models_initialized = True
        print("✅ All models initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize models: {e}")
        import traceback
        traceback.print_exc()
        raise

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    if not models_initialized:
        init_models()
    yield
    # Shutdown (cleanup if needed)
    clear_memory()

# Update app with lifespan
app = FastAPI(title="Make-It-Animatable API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models_initialized else "initializing", 
        "models_loaded": models_initialized,
        "device": str(device) if device else "unknown",
        "message": "Make-It-Animatable API is running"
    }

# Pipeline functions - adapted from app.py without gradio dependencies

def get_masked_mesh(mesh: trimesh.Trimesh, mask: np.ndarray):
    """Get masked mesh (from app.py)"""
    if mask is None:
        return mesh
    mesh = mesh.copy()
    if isinstance(mesh, trimesh.PointCloud):
        mesh.vertices = mesh.vertices[mask]
        mesh.colors = mesh.colors[mask]
    else:
        mesh.update_vertices(mask)
    return mesh

def prepare_input(input_path: str, is_gs=False, opacity_threshold=0.0, db: DB = None, export_temp=False):
    """Prepare input (adapted from app.py)"""
    if not (input_path and os.path.isfile(input_path)):
        raise RuntimeError(f"Input file not found: '{input_path}'")

    ply_path = f"{os.path.splitext(input_path)[0]}.ply"
    if os.path.isfile(ply_path):
        input_path = ply_path
    print(f"{input_path=}")

    if is_gs:
        if not input_path.endswith(".ply"):
            raise RuntimeError("Input must be a `.ply` file for Gaussian Splats")
        try:
            gaussians = load_gs(input_path)
            db.gs = gaussians
        except:
            raise RuntimeError("Fail to load the input file as Gaussian Splats")
        xyz, opacities, scales, rots, shs = gaussians.split((3, 1, 3, 4, 3), dim=-1)
        verts = xyz.numpy().astype(np.float32)
        sample_mask = (opacities >= opacity_threshold).squeeze(-1).numpy()
        assert sample_mask.any(), "No solid points"
        colors = shs.numpy().astype(np.float32)
        faces = None
        mesh = trimesh.PointCloud(verts, colors=colors, process=False)
    else:
        mesh: trimesh.Trimesh = trimesh.load(input_path, force="mesh")
        verts = np.array(mesh.vertices).astype(np.float32)
        sample_mask = None
        if isinstance(mesh, trimesh.PointCloud):
            faces = None
        else:
            verts_normal = np.array(mesh.vertex_normals).astype(np.float32)
            faces = np.array(mesh.faces)
    is_mesh = faces is not None
    pts = sample_mesh(get_masked_mesh(mesh, sample_mask), N, get_normals=is_mesh).astype(np.float32)
    pts = torch.from_numpy(pts).unsqueeze(0)
    verts = torch.from_numpy(verts).unsqueeze(0)
    if is_mesh:
        verts_normal = torch.from_numpy(verts_normal).unsqueeze(0)
        pts, pts_normal = torch.chunk(pts, 2, dim=-1)
    else:
        verts_normal = None
        pts_normal = None

    db.mesh = mesh
    db.is_mesh = is_mesh
    db.sample_mask = sample_mask
    db.verts = verts
    db.verts_normal = verts_normal
    db.faces = faces
    db.pts = pts
    db.pts_normal = pts_normal

    if export_temp:
        # Use test directory for debugging instead of temp
        output_dir = os.path.join(os.path.dirname(__file__), "test")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(output_dir, exist_ok=True)
    db.output_dir = output_dir
    db.joints_coarse_path = os.path.join(output_dir, "joints_coarse.glb")
    db.normed_path = os.path.join(output_dir, f"normed{os.path.splitext(input_path)[-1]}")
    db.sample_path = os.path.join(output_dir, "sample.glb")
    db.bw_path = os.path.join(output_dir, "bw.glb")
    db.joints_path = os.path.join(output_dir, "joints.glb")
    db.rest_lbs_path = os.path.join(output_dir, f"rest_lbs.{'ply' if is_gs else 'glb'}")
    db.rest_vis_path = os.path.join(output_dir, "rest.glb")
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    # For FBX output (animatable with rigging), use .fbx extension
    db.anim_path = os.path.join(output_dir, f"{input_filename}_animated.{'blend' if is_gs else 'fbx'}")
    db.anim_vis_path = os.path.join(output_dir, f"{input_filename}_animated.glb")

    return {"state": db}

@torch.no_grad()
def model_forward_coarse(pts: torch.Tensor) -> torch.Tensor:
    """Forward pass for coarse model"""
    pts = pts.to(device)
    joints = model_coarse(pts).joints
    return joints.cpu()

def preprocess(db: DB):
    """Preprocess step (adapted from app.py)"""
    mesh = db.mesh
    pts = db.pts
    pts_normal = db.pts_normal
    verts = db.verts
    verts_normal = db.verts_normal

    # Transform to Hips coordinates
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=True)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)
    
    print("🔍 Running coarse joint localization...")
    joints = model_forward_coarse(pts)
    joints, joints_tail = joints[..., :3], joints[..., 3:]
    hips = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Hips"]]
    rightupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightUpLeg"]]
    leftupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftUpLeg"]]
    
    rotate = Transform3d(matrix=get_hips_transform(hips, rightupleg, leftupleg).transpose(-1, -2))
    global_transform = norm.compose(rotate)
    pts = rotate.transform_points(pts)
    verts = rotate.transform_points(verts)
    if db.is_mesh:
        import torch.nn.functional as F
        pts_normal = F.normalize(rotate.transform_normals(pts_normal), dim=-1)
        verts_normal = F.normalize(rotate.transform_normals(verts_normal), dim=-1)

    # Update mesh vertices for export
    mesh.vertices = verts.squeeze(0).cpu().numpy()
    if db.gs is not None:
        db.gs = transform_gs(db.gs, global_transform)
        save_gs(db.gs, db.normed_path)
    else:
        mesh.export(db.normed_path)

    # Handle hands resampling if needed
    global hands_resample_ratio, geo_resample_ratio
    if hands_resample_ratio > 0:
        joints_tail_hips = rotate.transform_points(joints_tail).squeeze(0).cpu().numpy()
        hands_centers = [
            joints_tail_hips[BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftHand"]],
            joints_tail_hips[BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightHand"]],
        ]
        pts = sample_mesh(
            get_masked_mesh(mesh, db.sample_mask),
            N,
            get_normals=db.is_mesh,
            attn_ratio=hands_resample_ratio,
            attn_centers=hands_centers,
            attn_geo_ratio=geo_resample_ratio,
        ).astype(np.float32)
    else:
        pts = pts.squeeze(0).cpu().numpy()
        if db.is_mesh:
            pts_normal = pts_normal.squeeze(0).cpu().numpy()
            pts = np.concatenate([pts, pts_normal], axis=-1)

    pts = torch.from_numpy(pts).unsqueeze(0)
    if db.is_mesh:
        pts, pts_normal = torch.chunk(pts, 2, dim=-1)

    db.verts = verts
    db.verts_normal = verts_normal
    db.pts = pts
    db.pts_normal = pts_normal
    db.global_transform = global_transform

    return {"state": db}

@torch.no_grad()
def model_forward_bw(
    verts: torch.Tensor, verts_normal: torch.Tensor, pts: torch.Tensor, pts_normal: torch.Tensor, input_normal: bool
) -> torch.Tensor:
    """Forward pass for blend weights models"""
    print(f"🔍 DEBUG - BW inference starting...")
    print(f"    Input shapes - verts: {verts.shape}, pts: {pts.shape}")
    print(f"    Input normals: {input_normal}, verts_normal: {'Yes' if verts_normal is not None else 'No'}")
    print(f"    Model device: {next(model_bw.parameters()).device}")
    
    forward_start = time.time()
    model_device = next(model_bw.parameters()).device
    pts = pts.to(model_device)
    pts_normal = None if pts_normal is None else pts_normal.to(model_device)

    CHUNK = 100000  # prevent OOM for high-res models
    bw = []
    verts_chunks = torch.split(verts, CHUNK, dim=-2)
    verts_normal_chunks = (
        ([None] * len(verts_chunks)) if verts_normal is None else torch.split(verts_normal, CHUNK, dim=-2)
    )
    
    print(f"    Processing {len(verts_chunks)} chunks...")
    for i, (verts_, verts_normal_) in enumerate(zip(verts_chunks, verts_normal_chunks)):
        chunk_start = time.time()
        verts_ = verts_.to(model_device)
        verts_normal_ = None if verts_normal_ is None else verts_normal_.to(model_device)
        
        print(f"    Chunk {i+1}/{len(verts_chunks)} - calling model_bw...")
        bw_ = model_bw(pts, verts_).bw
        print(f"    Chunk {i+1} - model_bw output shape: {bw_.shape}")
        
        if input_normal and model_bw_normal is not None:
            print(f"    Chunk {i+1} - calling model_bw_normal...")
            bw_normal = model_bw_normal(
                torch.cat([pts, pts_normal], dim=-1), torch.cat([verts_, verts_normal_], dim=-1)
            ).bw
            # Simple conflict resolution - in production would use get_conflict_mask
            mask = torch.argmax(bw_, dim=-1)
            spine_shoulder_arm_indices = [i for i, name in enumerate(bones_idx_dict_bw.keys()) 
                                        if any(x in name for x in ("Spine", "Shoulder", "Arm"))]
            if spine_shoulder_arm_indices:
                spine_mask = torch.isin(mask, torch.tensor(spine_shoulder_arm_indices, device=mask.device))
                bw_normal[spine_mask.unsqueeze(-1).expand_as(bw_normal)] = bw_[spine_mask.unsqueeze(-1).expand_as(bw_)]
            bw_ = bw_normal
            print(f"    Chunk {i+1} - normal processing complete")
        
        bw.append(bw_)
        print(f"    Chunk {i+1} completed in {time.time() - chunk_start:.3f}s")
    
    bw = torch.cat(bw, dim=-2)
    forward_time = time.time() - forward_start
    print(f"🔍 DEBUG - BW inference completed in {forward_time:.3f}s")
    print(f"    Final BW shape: {bw.shape}, device: {bw.device}")
    print(f"    BW value range: [{bw.min().item():.6f}, {bw.max().item():.6f}]")

    return bw.cpu()

@torch.no_grad()
def model_forward_bones(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for joints and pose models"""
    print(f"🔍 DEBUG - Bones inference starting...")
    print(f"    Input pts shape: {pts.shape}")
    print(f"    Device: {device}")
    
    forward_start = time.time()
    pts = pts.to(device)

    # Joints inference
    joints_start = time.time()
    print(f"    Calling model_joints...")
    joints = model_joints.forward(pts).joints
    print(f"    Joints inference completed in {time.time() - joints_start:.3f}s")
    print(f"    Joints shape: {joints.shape}")
    print(f"    Joints value range: [{joints.min().item():.6f}, {joints.max().item():.6f}]")
    
    if joints_additional and model_joints_add is not None:
        print(f"    Calling additional joints model...")
        joints_add = model_joints_add(pts).joints
        print(f"    Additional joints shape: {joints_add.shape}")
        # In simplified version, just use the main joints
        pass

    # Pose inference
    pose_start = time.time()
    if model_pose.pose_input_joints:
        joints_ = joints.clone()
        if joints_additional:
            # Simplified - use joints as is
            pass
    else:
        joints_ = None
    
    print(f"    Calling model_pose...")
    pose = model_pose(pts, joints=joints_).pose_trans
    print(f"    Pose inference completed in {time.time() - pose_start:.3f}s")
    print(f"    Pose shape: {pose.shape}")
    print(f"    Pose value range: [{pose.min().item():.6f}, {pose.max().item():.6f}]")
    
    forward_time = time.time() - forward_start
    print(f"🔍 DEBUG - Bones inference completed in {forward_time:.3f}s")

    return joints.cpu(), pose.cpu()

def infer(input_normal: bool, db: DB):
    """Inference step - full implementation"""
    print("🧠 Running model inference...")
    
    pts = db.pts
    pts_normal = db.pts_normal
    verts = db.verts
    verts_normal = db.verts_normal

    if input_normal and not db.is_mesh:
        raise RuntimeError("Normals are not available for point clouds or Gaussian Splats")

    # Normalize data & infer the main model
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=False)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)
    
    if db.gs is not None:
        db.gs = transform_gs(db.gs, norm)

    print("🔍 Running blend weights inference...")
    bw = model_forward_bw(verts, verts_normal, pts, pts_normal, input_normal)
    
    print("🔍 Running joints and pose inference...")
    joints, pose = model_forward_bones(pts)

    db.mesh.vertices = verts.squeeze(0).cpu().numpy()
    db.pts = pts
    db.verts = verts
    db.bw = bw
    db.joints = joints
    db.pose = pose
    db.global_transform = db.global_transform.compose(norm)
    
    print("✅ Model inference completed")
    return {"state": db}

def vis(bw_fix: bool, bw_vis_bone: str, no_fingers: bool, db: DB):
    """Visualization step - adapted from app.py with full LBS implementation"""
    print("🎨 Creating visualizations...")
    
    verts = db.verts
    bw = db.bw
    joints = db.joints
    pose = db.pose
    
    # Extract joints data properly
    if joints.shape[-1] >= 6:  # joints + tail
        joints_pos = joints.squeeze(0)[..., :3].cpu().numpy()
        joints_tail_data = joints.squeeze(0)[..., 3:6].cpu().numpy()
    else:
        joints_pos = joints.squeeze(0)[..., :3].cpu().numpy()
        joints_tail_data = joints_pos  # fallback
    
    # Process blend weights with full post-processing from app.py
    if bw_fix:
        print("🔧 Post-processing blend weights...")
        # Keep blend weights normalized
        bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)
        # Only keep weights from the largest-weighted joints
        joints_per_point = 4
        thresholds = torch.topk(bw, k=joints_per_point, dim=-1, sorted=True).values[..., -1:]
        bw[bw < thresholds] = 0
        bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Convert to numpy
    bw_np = bw.squeeze(0).cpu().numpy()
    verts_np = verts.squeeze(0).cpu().numpy()
    
    print(f"📊 Processing results:")
    print(f"   - Vertices: {verts_np.shape}")
    print(f"   - Blend weights: {bw_np.shape}")
    print(f"   - Joints: {joints_pos.shape}")
    print(f"   - Pose: {pose.shape if pose is not None else 'None'}")
    
    try:
        # Create visualizations using trimesh directly (like app.py)
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap("plasma")
        
        # Blend weights visualization
        bw_vis_bone_idx = bones_idx_dict_bw.get(f"{MIXAMO_PREFIX}{bw_vis_bone}", 0)
        colors = cmap(bw_np[:, bw_vis_bone_idx])[:, :3]
        
        if db.faces is None:
            bw_mesh = trimesh.PointCloud(verts_np, process=False, colors=colors)
        else:
            bw_mesh = trimesh.Trimesh(verts_np, db.faces, process=False, maintain_order=True, vertex_colors=colors)
        
        extent = verts_np.max() - verts_np.min()
        axis = trimesh.creation.axis(origin_size=extent * 0.02)
        bw_scene = trimesh.Scene()
        bw_scene.add_geometry(bw_mesh, geom_name="mesh")
        bw_scene.add_geometry(axis, geom_name="axis")
        bw_scene.export(db.bw_path)
        print(f"📁 Blend weights visualization: {db.bw_path}")
        
        # Joints visualization
        if db.faces is None:
            colors_joint = cmap(np.linalg.norm(joints_pos[0] - verts_np, axis=-1))[:, :3]
            joints_mesh = trimesh.PointCloud(verts_np, process=False, colors=colors_joint)
        else:
            normals = trimesh.Trimesh(verts_np, db.faces, process=False, maintain_order=True).vertex_normals
            colors_joint = (normals + 1) / 2
            joints_mesh = trimesh.Trimesh(verts_np, db.faces, process=False, maintain_order=True, vertex_colors=colors_joint)
        
        scene = trimesh.Scene()
        scene.add_geometry(joints_mesh, geom_name="mesh")
        
        # Add joint markers
        for joint, joint_name in zip(joints_pos, bones_idx_dict_joints):
            if "Hand" in joint_name:
                scaling = 0.01
            elif "Hips" in joint_name or "Spine" in joint_name:
                scaling = 0.06
            else:
                scaling = 0.04
            
            marker = trimesh.creation.icosphere(radius=extent * scaling)
            marker.vertices = marker.vertices + joint
            scene.add_geometry(marker, geom_name=joint_name)
        
        axis = trimesh.creation.axis(origin_size=extent * 0.02)
        scene.add_geometry(axis, geom_name="axis")
        scene.export(db.joints_path)
        print(f"📁 Joints visualization: {db.joints_path}")
        
        # Rest pose with proper LBS (adapted from app.py)
        if pose is not None:
            print("🔧 Processing pose for rest pose visualization...")
            
            # Handle different pose modes (from app.py)
            if joints_additional:
                # Reorganize bone data if needed
                pass
            
            if "local" in model_pose.pose_mode:
                pose = to_pose_local(pose, input_mode=model_pose.pose_mode, return_quat=False)
                pose, _ = pose_local_to_global(
                    pose,
                    db.joints[..., :3],
                    torch.tensor((KINEMATIC_TREE_ADD if joints_additional else KINEMATIC_TREE).parent_indices),
                    relative_to_source=True,
                )
            elif model_pose.pose_mode in ("quat", "ortho6d"):
                pose, _ = pose_rot_to_global(
                    pose,
                    db.joints[..., :3],
                    torch.tensor((KINEMATIC_TREE_ADD if joints_additional else KINEMATIC_TREE).parent_indices),
                )
            else:
                pose = to_pose_matrix(pose, input_mode=model_pose.pose_mode, source=db.joints[..., :3])
            
            # Set identity for root bone
            pose[..., 0, :, :] = torch.eye(4)
            pose_matrices = pose.squeeze(0).cpu().numpy()
            
            # Apply LBS transformation (from app.py)
            lbs_transform = np.einsum("kij,nk->nij", pose_matrices, bw_np)
            
            if db.gs is None:
                # Apply transformation to vertices
                rest_joints = apply_transform(joints_pos, pose_matrices)
                
                # Create visualization with transformed vertices and joints
                transformed_verts = apply_transform(verts_np, lbs_transform)
                
                # Create rest pose visualization
                if db.faces is None:
                    colors_rest = cmap(np.linalg.norm(rest_joints[0] - transformed_verts, axis=-1))[:, :3]
                    rest_vis_mesh = trimesh.PointCloud(transformed_verts, process=False, colors=colors_rest)
                else:
                    normals = trimesh.Trimesh(transformed_verts, db.faces, process=False, maintain_order=True).vertex_normals
                    colors_rest = (normals + 1) / 2
                    rest_vis_mesh = trimesh.Trimesh(transformed_verts, db.faces, process=False, maintain_order=True, vertex_colors=colors_rest)
                
                scene = trimesh.Scene()
                scene.add_geometry(rest_vis_mesh, geom_name="mesh")
                
                # Add transformed joint markers
                for joint, joint_name in zip(rest_joints, bones_idx_dict_joints):
                    if "Hand" in joint_name:
                        scaling = 0.01
                    elif "Hips" in joint_name or "Spine" in joint_name:
                        scaling = 0.06
                    else:
                        scaling = 0.04
                    
                    marker = trimesh.creation.icosphere(radius=extent * scaling)
                    marker.vertices = marker.vertices + joint
                    scene.add_geometry(marker, geom_name=joint_name)
                
                axis = trimesh.creation.axis(origin_size=extent * 0.02)
                scene.add_geometry(axis, geom_name="axis")
                scene.export(db.rest_lbs_path)
                print(f"📁 Rest pose with LBS: {db.rest_lbs_path}")
            else:
                # For Gaussian Splats
                db.gs_rest = transform_gs(db.gs, lbs_transform)
                save_gs(db.gs_rest, db.rest_lbs_path)
                print(f"📁 Rest pose GS with LBS: {db.rest_lbs_path}")
        else:
            # No pose - just copy original mesh
            db.mesh.export(db.rest_lbs_path)
            print(f"📁 Rest pose (original): {db.rest_lbs_path}")
            
    except Exception as e:
        print(f"⚠️ Visualization export error: {e}")
        import traceback
        traceback.print_exc()
        # Create fallback - just copy the mesh
        try:
            db.mesh.export(db.bw_path)
            db.mesh.export(db.joints_path)
            db.mesh.export(db.rest_lbs_path)
        except:
            pass
    
    # Update database
    db.verts = verts
    db.bw = bw
    db.joints = joints
    db.joints_tail = joints_tail_data
    db.pose = pose
    
    print("✅ Visualizations created")
    return {"state": db}

def get_pose_ignore_list(pose: str = None, pose_parts: list[str] = None):
    """Get pose ignore list (from app.py)"""
    kw_list: list[str] = ["Hips", "Ear", "Tail"]
    if pose:
        if pose == "T-pose":
            kw_list.extend([
                "Spine", "Neck", "Head", "Shoulder", "Arm", "ForeArm", "Hand",
                "UpLeg", "Leg", "Foot", "ToeBase",
            ])  # all
        elif pose == "A-pose":
            kw_list.extend([
                "Spine", "Neck", "Head", "ForeArm", "Hand",
                "UpLeg", "Leg", "Foot", "ToeBase",
            ])  # except for Shoulder & Arm
        elif pose == "大-pose":
            kw_list.extend([
                "Spine", "Neck", "Head", "Shoulder", "Arm", "ForeArm", "Hand",
                "Leg", "Foot", "ToeBase",
            ])  # except for UpLeg
    if pose_parts:
        if "Fingers" in pose_parts:
            kw_list.extend(["Thumb", "Index", "Middle", "Ring", "Pinky"])
        if "Arms" in pose_parts:
            kw_list.extend(["Arm", "ForeArm", "Hand"])
        if "Legs" in pose_parts:
            kw_list.extend(["UpLeg", "Leg", "Foot", "ToeBase"])
        if "Head" in pose_parts:
            kw_list.extend(["Head", "Neck"])
    return kw_list

def vis_blender(
    reset_to_rest: bool,
    no_fingers: bool,
    rest_pose_type: str,
    ignore_pose_parts: list,
    animation_file: str,
    retarget: bool,
    inplace: bool,
    db: DB,
):
    """Blender visualization - create actual rigged FBX using app_blender.py (app.py compatible)"""
    print("🎬 Creating final animation with Blender rigging...")
    print(f"    Animation file: {animation_file if animation_file else 'None'}")
    print(f"    Output path: {db.anim_path}")
    
    if any(x is None for x in (db.mesh, db.joints, db.joints_tail, db.bw)):
        raise RuntimeError("Missing inference data - run inference first")
    
    if db.gs is not None:
        print("⚠️ Gaussian Splats processing - may take longer in Blender")
        if isinstance(db.gs, torch.Tensor):
            db.gs = db.gs.numpy()
        if hasattr(db, 'gs_rest') and db.gs_rest is not None and isinstance(db.gs_rest, torch.Tensor):
            db.gs_rest = db.gs_rest.numpy()
    
    template_path = TEMPLATE_PATH_ADD if joints_additional else TEMPLATE_PATH
    
    # Prepare data exactly like app.py - ensure correct numpy format and shapes
    # Convert all tensors to numpy with correct shapes
    if isinstance(db.joints, torch.Tensor):
        joints_np = db.joints.squeeze(0).cpu().numpy()  # Remove batch dimension: [1, 52, 6] -> [52, 6]
    else:
        joints_np = np.array(db.joints).squeeze(0) if hasattr(db.joints, 'squeeze') else np.array(db.joints)
    
    # Extract head and tail positions properly
    if joints_np.shape[-1] >= 6:
        joints_head = joints_np[..., :3]  # [52, 3]
        joints_tail_data = joints_np[..., 3:6]  # [52, 3] 
    else:
        joints_head = joints_np[..., :3]  # [52, 3]
        joints_tail_data = joints_head  # fallback
    
    # Ensure we have joints_tail data
    if hasattr(db, 'joints_tail') and db.joints_tail is not None:
        if isinstance(db.joints_tail, torch.Tensor):
            joints_tail_data = db.joints_tail.squeeze(0).cpu().numpy() if db.joints_tail.dim() > 2 else db.joints_tail.cpu().numpy()
        else:
            joints_tail_data = np.array(db.joints_tail)
    
    # Convert blend weights
    if isinstance(db.bw, torch.Tensor):
        bw_np = db.bw.squeeze(0).cpu().numpy()  # Remove batch dimension: [1, N, 52] -> [N, 52]
    else:
        bw_np = np.array(db.bw).squeeze(0) if hasattr(db.bw, 'squeeze') else np.array(db.bw)
    
    # Convert pose
    if db.pose is not None:
        if isinstance(db.pose, torch.Tensor):
            pose_np = db.pose.squeeze(0).cpu().numpy()  # Remove batch dimension: [1, 52, 6] -> [52, 6]
        else:
            pose_np = np.array(db.pose).squeeze(0) if hasattr(db.pose, 'squeeze') else np.array(db.pose)
    else:
        pose_np = None
    
    # Debug: Print shapes for verification
    print(f"📊 DEBUG - Data shapes for app_blender.py:")
    print(f"    joints_head: {joints_head.shape}")
    print(f"    joints_tail: {joints_tail_data.shape}")
    print(f"    bw: {bw_np.shape}")
    print(f"    pose: {pose_np.shape if pose_np is not None else 'None'}")
    print(f"    bones_idx_dict length: {len(bones_idx_dict_joints)}")
    print(f"    Expected: joints.shape[0] == len(bones_idx_dict): {joints_head.shape[0]} == {len(bones_idx_dict_joints)}")
    
    # Verify shapes match expectations
    if joints_head.shape[0] != len(bones_idx_dict_joints):
        raise RuntimeError(f"Joints shape mismatch: joints.shape[0]={joints_head.shape[0]} != len(bones_idx_dict)={len(bones_idx_dict_joints)}")
    
    data = dict(
        mesh=db.mesh,
        gs=db.gs_rest if hasattr(db, 'gs_rest') and reset_to_rest else db.gs,
        joints=joints_head,  # [52, 3] - head positions only
        joints_tail=joints_tail_data,  # [52, 3] - tail positions  
        bw=bw_np,  # [N, 52] - blend weights
        pose=pose_np,  # [52, 6] or [52, 4, 4] - pose data
        bones_idx_dict=dict(bones_idx_dict_joints),
        pose_ignore_list=get_pose_ignore_list(rest_pose_type, ignore_pose_parts),
    )
    
    if animation_file is not None:
        if not os.path.isfile(animation_file):
            raise RuntimeError(f"Animation file {animation_file} does not exist")
        if not reset_to_rest:
            print("⚠️ 'Reset to Rest' is not enabled, animation may be incorrect if input is not in T-pose")
    
    try:
        # Call app_blender.py exactly like app.py does
        # Use subprocess approach since we're in API context (not main thread)
        print("🔥 Calling app_blender.py for Blender rigging...")
        
        # Save data to temporary npz file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **data)
            temp_data_path = f.name
        
        try:
            # Try direct import approach first (like app.py main thread)
            try:
                print("    Attempting direct app_blender import...")
                parent_dir = os.path.dirname(__file__) + "/.."
                sys.path.insert(0, os.path.abspath(parent_dir))
                
                from argparse import Namespace
                from app_blender import main as app_blender_main
                
                # Call directly like app.py does in main thread
                print(f"    Calling app_blender_main with data shapes verified")
                app_blender_main(
                    Namespace(
                        input_path=data,  # Pass data directly, not file path
                        output_path=db.anim_path,
                        template_path=template_path,
                        keep_raw=False,
                        rest_path=db.rest_vis_path if db.is_mesh else None,
                        pose_local=False,
                        reset_to_rest=reset_to_rest,
                        remove_fingers=no_fingers,
                        animation_path=animation_file,
                        retarget=retarget,
                        inplace=inplace,
                    )
                )
                print("    Direct app_blender call succeeded")
                
            except Exception as direct_error:
                print(f"    Direct import failed: {direct_error}")
                print("    Falling back to subprocess approach...")
                
                # Fallback to subprocess (app.py child thread style)
                original_cwd = os.getcwd()
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                os.chdir(parent_dir)
                
                try:
                    cmd = f"python app_blender.py --input_path '{temp_data_path}' --output_path '{os.path.abspath(db.anim_path)}'"
                    cmd += f" --template_path '{os.path.abspath(template_path)}'"
                    
                    if db.is_mesh and hasattr(db, 'rest_vis_path'):
                        cmd += f" --rest_path '{os.path.abspath(db.rest_vis_path)}'"
                    
                    if reset_to_rest:
                        cmd += " --reset_to_rest"
                    if no_fingers:
                        cmd += " --remove_fingers"
                    if animation_file is not None:
                        cmd += f" --animation_path '{os.path.abspath(animation_file)}'"
                        if retarget:
                            cmd += " --retarget"
                        if inplace:
                            cmd += " --inplace"
                    
                    print(f"    Executing from {os.getcwd()}: {cmd}")
                    
                    # Remove output suppression for debugging
                    result = os.system(cmd)
                    
                    if result != 0:
                        raise RuntimeError(f"app_blender.py subprocess failed with exit code {result}")
                        
                finally:
                    os.chdir(original_cwd)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_data_path):
                os.unlink(temp_data_path)
        
        print(f"📁 Output animatable model: '{db.anim_path}'")
        
        # Verify the file was actually created
        if not os.path.exists(db.anim_path):
            # Look for files with similar names in the output directory
            output_dir = os.path.dirname(db.anim_path)
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                animated_files = [f for f in files if '_animated' in f and f.endswith(('.fbx', '.glb', '.blend'))]
                if animated_files:
                    # Use the first found animated file
                    actual_path = os.path.join(output_dir, animated_files[0])
                    print(f"⚠️ Expected file not found, using: '{actual_path}'")
                    db.anim_path = actual_path
                else:
                    raise RuntimeError(f"Expected animated file not found at {db.anim_path}")
        
        # FBX to GLB conversion for preview (like app.py)
        if db.is_mesh and db.anim_path.endswith(".fbx") and os.path.isfile(db.anim_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Use FBX2glTF converter like app.py
                fbx2glb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "util", "FBX2glTF")
                if os.path.isfile(fbx2glb_path):
                    fbx2glb_cmd = f"{fbx2glb_path} --binary --keep-attribute auto --fbx-temp-dir '{tmpdir}' --input '{os.path.abspath(db.anim_path)}' --output '{os.path.abspath(db.anim_vis_path)}'"
                    fbx2glb_cmd += " > /dev/null 2>&1"
                    os.system(fbx2glb_cmd)
                    if os.path.exists(db.anim_vis_path):
                        print(f"📁 Output visualization: '{db.anim_vis_path}'")
                else:
                    print(f"⚠️ FBX2glTF converter not found at {fbx2glb_path}")
                    db.anim_vis_path = None
        else:
            db.rest_vis_path = None
            db.anim_vis_path = None
        
        # No compression - always return the original file
        # db.anim_path remains unchanged
        
    except Exception as e:
        print(f"❌ Blender processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Blender rigging failed: {e}")
    
    return {"state": db}

@torch.no_grad()
def run_pipeline(
    input_path: str,
    is_gs: bool = False,
    opacity_threshold: float = 0.0,
    no_fingers: bool = False,
    rest_pose_type: str = None,
    ignore_pose_parts: List[str] = None,
    input_normal: bool = False,
    bw_fix: bool = True,
    bw_vis_bone: str = "LeftArm",
    reset_to_rest: bool = False,
    animation_file: str = None,
    retarget: bool = True,
    inplace: bool = True,
    export_temp: bool = True,
) -> str:
    """
    Run the complete Make-It-Animatable pipeline
    Returns the path to the final animated model
    """
    db = DB()
    
    try:
        start_time = time.time()
        with TimePrints():
            print("*" * 50)
            print(f"🎯 Processing: {input_path}")
            print(f"🕐 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        clear_memory(db)
        
        # Step 1: Prepare input
        step_start = time.time()
        print("📥 Preparing input...")
        result = prepare_input(input_path, is_gs, opacity_threshold, db, export_temp)
        db = result['state'] if isinstance(result, dict) and 'state' in result else db
        print(f"⏱️  Prepare input completed in {time.time() - step_start:.2f}s")
        print(f"📊 Input data - Verts: {db.verts.shape if db.verts is not None else 'None'}, Pts: {db.pts.shape if db.pts is not None else 'None'}")
        
        # Step 2: Preprocess
        step_start = time.time()
        print("🔄 Preprocessing...")
        result = preprocess(db)
        db = result['state'] if isinstance(result, dict) and 'state' in result else db
        print(f"⏱️  Preprocessing completed in {time.time() - step_start:.2f}s")
        
        # Step 3: Inference
        step_start = time.time()
        print("🧠 Running inference...")
        result = infer(input_normal, db)
        db = result['state'] if isinstance(result, dict) and 'state' in result else db
        print(f"⏱️  Inference completed in {time.time() - step_start:.2f}s")
        print(f"📊 Inference results - BW: {db.bw.shape if db.bw is not None else 'None'}, Joints: {db.joints.shape if db.joints is not None else 'None'}")
        
        # Step 4: Visualization
        step_start = time.time()
        print("🎨 Generating visualizations...")
        result = vis(bw_fix, bw_vis_bone, no_fingers, db)
        db = result['state'] if isinstance(result, dict) and 'state' in result else db
        print(f"⏱️  Visualization completed in {time.time() - step_start:.2f}s")
        
        # Step 5: Blender processing
        step_start = time.time()
        print("🎬 Creating animation...")
        result = vis_blender(
            reset_to_rest, no_fingers, rest_pose_type, ignore_pose_parts or [], 
            animation_file, retarget, inplace, db
        )
        db = result['state'] if isinstance(result, dict) and 'state' in result else db
        print(f"⏱️  Blender processing completed in {time.time() - step_start:.2f}s")
        
        # Use anim_path as final output (FBX or compressed zip)
        output_path = db.anim_path
            
        if not output_path or not os.path.exists(output_path):
            raise RuntimeError("Animation processing failed - no output file generated")
        
        total_time = time.time() - start_time
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        print(f"✅ Animation complete! Total time: {total_time:.2f}s")
        print(f"📁 Output: {output_path} ({output_size} bytes)")
        print(f"🕐 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        clear_memory(db)

@app.post("/animate")
async def animate_character(
    background_tasks: BackgroundTasks,
    input_file: UploadFile = File(...),
    animation_file: Optional[UploadFile] = File(None),
    # Form data for parameters
    is_gs: bool = Form(False),
    opacity_threshold: float = Form(0.01),
    no_fingers: bool = Form(True),
    rest_pose_type: str = Form("No"),
    rest_parts: str = Form("[]"),  # JSON string
    input_normal: bool = Form(False),
    bw_fix: bool = Form(True),
    bw_vis_bone: str = Form("LeftArm"),
    reset_to_rest: bool = Form(True),
    retarget: bool = Form(True),
    inplace: bool = Form(True),
):
    """
    Main endpoint to animate a 3D character
    """
    if not models_initialized:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    # Create temporary directory for this request
    request_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / f"mia_request_{request_id}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Parse rest_parts JSON
        try:
            rest_parts_list = json.loads(rest_parts) if rest_parts else []
        except json.JSONDecodeError:
            rest_parts_list = []
        
        # Save uploaded input file
        input_suffix = Path(input_file.filename).suffix if input_file.filename else '.glb'
        input_path = temp_dir / f"input{input_suffix}"
        with open(input_path, "wb") as f:
            content = await input_file.read()
            f.write(content)
        
        # Save animation file if provided
        animation_path = None
        if animation_file:
            anim_suffix = Path(animation_file.filename).suffix if animation_file.filename else '.fbx'
            animation_path = temp_dir / f"animation{anim_suffix}"
            with open(animation_path, "wb") as f:
                content = await animation_file.read()
                f.write(content)
        
        # Run pipeline
        print(f"🎯 Processing animation request {request_id}")
        output_path = run_pipeline(
            input_path=str(input_path),
            is_gs=is_gs,
            opacity_threshold=opacity_threshold,
            no_fingers=no_fingers,
            rest_pose_type=rest_pose_type,
            ignore_pose_parts=rest_parts_list,
            input_normal=input_normal,
            bw_fix=bw_fix,
            bw_vis_bone=bw_vis_bone,
            reset_to_rest=reset_to_rest,
            animation_file=str(animation_path) if animation_path else None,
            retarget=retarget,
            inplace=inplace,
            export_temp=True
        )
        
        # Determine output filename - use GLB instead of FBX
        base_name = Path(input_file.filename).stem if input_file.filename else "animated"
        # Check actual output file extension (may have been changed from fbx to glb)
        output_ext = Path(output_path).suffix.lstrip('.')
        filename = f"{base_name}_animated.{output_ext}"
        
        # Schedule cleanup
        def cleanup():
            import shutil
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Cleanup warning: {e}")
        
        background_tasks.add_task(cleanup)
        
        # Return the animated file
        return FileResponse(
            path=output_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        # Cleanup on error
        import shutil
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass
        
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/models/status")
async def models_status():
    """Get detailed model status"""
    return {
        "initialized": models_initialized,
        "device": str(device) if device else None,
        "models": {
            "bw": model_bw is not None,
            "bw_normal": model_bw_normal is not None,
            "joints": model_joints is not None,
            "joints_add": model_joints_add is not None,
            "coarse": model_coarse is not None,
            "pose": model_pose is not None,
        },
        "config": {
            "N": N,
            "hands_resample_ratio": hands_resample_ratio,
            "geo_resample_ratio": geo_resample_ratio,
            "bw_additional": bw_additional,
            "joints_additional": joints_additional,
        }
    }

@app.get("/dev/info")
async def dev_info():
    """Development info endpoint"""
    import psutil
    import torch
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory": {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
            }
        }
    
    return {
        "system": {
            "cwd": os.getcwd(),
            "python_path": sys.path[:5],  # First 5 paths
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "gpu_info": gpu_info,
        "models_loaded": models_initialized,
        "environment": "development" if uvicorn.__version__ else "production"
    }

@app.post("/dev/test_pipeline")
async def test_pipeline_step(
    input_file: UploadFile = File(...),
    step: str = Form("prepare_input")  # prepare_input, preprocess, infer, vis, vis_blender
):
    """Test individual pipeline step for debugging"""
    import tempfile
    import uuid
    from pathlib import Path
    
    # Create temp file
    request_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.gettempdir()) / f"mia_debug_{request_id}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save input
        input_path = temp_dir / f"input_{input_file.filename}"
        with open(input_path, "wb") as f:
            content = await input_file.read()
            f.write(content)
        
        db = DB()
        result = {"step": step, "status": "success"}
        
        if step == "prepare_input":
            prepare_input(str(input_path), db=db, export_temp=True)
            result["output"] = {
                "is_mesh": db.is_mesh,
                "verts_shape": list(db.verts.shape) if db.verts is not None else None,
                "pts_shape": list(db.pts.shape) if db.pts is not None else None,
                "output_dir": db.output_dir
            }
        
        return result
        
    except Exception as e:
        return {"step": step, "status": "error", "error": str(e)}
    finally:
        # Cleanup
        import shutil
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    # Run the server with hot reload
    print("🔥 Starting Make-It-Animatable API Server with hot reload...")
    print("📍 Working directory:", os.getcwd())
    print("🌐 Server will be available at: http://127.0.0.1:8765")
    print("📝 API docs at: http://127.0.0.1:8765/docs")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,  # Disable reload due to bpy conflicts
        log_level="info"
    )