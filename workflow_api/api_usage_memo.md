# 起動
```
cd workflow_api
source .venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

# generate_character
```
curl -X POST "http://localhost:8000/generate_character" \
  -F "image=@input/avatar_base.jpg" \
  -F "enable_rigging=true"
```

# 進捗確認
```
curl "http://localhost:8000/job/{JOB_ID}/status"
```

# ダウンロード
```
curl "http://localhost:8000/job/{job_id}/download/{filename}"
```