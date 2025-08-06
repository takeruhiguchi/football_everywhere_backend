```
cd /home/takeru.higuchi/TextureGeneration/ComfyUI/custom_nodes/Make-It-Animatable/
source .venv/bin/activate  # 仮想環境をアクティベート
cd api_server
python main.py
```

```
# ファイルエコーテスト（GLBファイルがある場合）
curl -X POST "http://127.0.0.1:8765/echo" \
-F "input_file=@test/test.glb" \
--output test/echo_output.glb
```

```
curl -X POST "http://127.0.0.1:8765/animate" \
-F "input_file=@test/test.glb" \
-F "no_fingers=true" \
-F "reset_to_rest=true" \
--output test/output_animated.fbx
```