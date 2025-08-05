# ComfyUI Texture Generation API Specification

## Base URL
```
http://localhost:8188
```

## Endpoints

### 1. Generate 3D Character with Texture

**POST** `/generate_character`

Creates a 3D character with texture from an input image.

#### Request

**Content-Type**: `multipart/form-data`

##### Form Parameters:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Input image file (PNG, JPG, JPEG) |
| `prompt` | String | No | "Highly Detailed" | Text prompt for texture generation |
| `negative_prompt` | String | No | "watermark, ugly, deformed, noisy, blurry, low contrast" | Negative prompt |
| `image_size` | Integer | No | 768 | Size for multiview generation (512, 768, 1024) |
| `mesh_input_size` | Integer | No | 1024 | Input image processing size |
| `outline_size` | Integer | No | 128 | Border outline thickness |
| `steps` | Integer | No | 10 | Generation steps (higher = better quality, slower) |
| `guidance_scale` | Float | No | 5.5 | How closely to follow prompt (1.0-20.0) |
| `cfg_scale` | Float | No | 2.0 | CFG scale for texture generation |
| `seed` | Integer | No | -1 | Random seed (-1 for random) |
| `model_name` | String | No | "dreamshaper.safetensors" | Base diffusion model |
| `pose_type` | String | No | "大-pose" | Rest pose type for rigging |
| `enable_rigging` | Boolean | No | true | Whether to generate rigged character |

##### Example Request:
```bash
curl -X POST http://localhost:8188/generate_character \
  -F "image=@input.jpg" \
  -F "prompt=anime character with detailed clothing" \
  -F "image_size=768" \
  -F "steps=15" \
  -F "seed=42"
```

#### Response

**Content-Type**: `application/json`

##### Success Response (202 Accepted):
```json
{
  "status": "accepted",
  "job_id": "uuid-string",
  "message": "Character generation started",
  "estimated_time": "120-180 seconds"
}
```

##### Error Response (400 Bad Request):
```json
{
  "status": "error",
  "error_code": "INVALID_PARAMETERS",
  "message": "Invalid image size. Must be 512, 768, or 1024",
  "details": {
    "parameter": "image_size",
    "provided": 999,
    "allowed": [512, 768, 1024]
  }
}
```

### 2. Check Job Status

**GET** `/job/{job_id}/status`

Check the status of a character generation job.

#### Response

##### In Progress:
```json
{
  "status": "processing",
  "job_id": "uuid-string",
  "progress": 45,
  "current_stage": "generating_multiview_texture",
  "stages": [
    {"name": "preprocessing", "status": "completed"},
    {"name": "mesh_generation", "status": "completed"},
    {"name": "generating_multiview_texture", "status": "processing"},
    {"name": "texture_baking", "status": "pending"},
    {"name": "rigging", "status": "pending"}
  ],
  "estimated_remaining": "75 seconds"
}
```

##### Completed:
```json
{
  "status": "completed",
  "job_id": "uuid-string",
  "progress": 100,
  "results": {
    "raw_mesh": "/output/3D/Hy3D_00001_.glb",
    "textured_mesh": "/output/3D/Hy3D_textured_00001_.glb",
    "rigged_character": "/output/3D/Hy3D_textured_00001__rigged.fbx",
    "preview_images": [
      "/output/preview_mesh.png",
      "/output/preview_texture.png"
    ]
  },
  "processing_time": "142 seconds"
}
```

##### Failed:
```json
{
  "status": "failed",
  "job_id": "uuid-string",
  "error": "Mesh generation failed",
  "error_details": "CUDA out of memory",
  "failed_stage": "mesh_generation"
}
```

### 3. Download Results

**GET** `/job/{job_id}/download/{filename}`

Download generated files.

#### Response
- **Content-Type**: `application/octet-stream`
- **Content-Disposition**: `attachment; filename="result.glb"`

### 4. List Available Models

**GET** `/models`

Get list of available models.

#### Response
```json
{
  "diffusion_models": [
    "dreamshaper.safetensors",
    "realistic_vision.safetensors"
  ],
  "3d_models": [
    "hunyuan3d-dit-v2-0-fp16.safetensors"
  ]
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_PARAMETERS` | Request parameters are invalid |
| `FILE_TOO_LARGE` | Uploaded image exceeds size limit |
| `UNSUPPORTED_FORMAT` | Image format not supported |
| `JOB_NOT_FOUND` | Job ID does not exist |
| `PROCESSING_ERROR` | Error during generation |
| `SERVER_BUSY` | Server at capacity, try again later |

## Rate Limits

- **Requests per minute**: 10
- **Concurrent jobs per user**: 2
- **Maximum file size**: 10MB

## WebSocket Events (Optional)

For real-time updates, connect to:
```
ws://localhost:8188/ws/{job_id}
```

Events:
- `progress_update`: Progress percentage and stage info
- `preview_ready`: Preview image available
- `stage_completed`: Individual stage completion
- `job_completed`: Final completion
- `error`: Error occurred