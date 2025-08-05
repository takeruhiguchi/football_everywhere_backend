# ComfyUI Texture Generation API

ComfyUIワークフローをAPI化して、3Dキャラクター生成とテクスチャリングを自動化するプロジェクトです。

## 機能

- 画像から3Dメッシュ生成 (Hunyuan3D)
- マルチビューテクスチャ生成 (IG2MV)
- テクスチャベイキングとインペインティング
- キャラクターリギング (Make-It-Animatable)
- RESTful API インターフェース
- リアルタイム進捗モニタリング

## セットアップ

### 1. 依存関係のインストール

```bash
cd /home/takeru.higuchi/TextureGeneration/workflow_api
pip install -r requirements.txt
```

### 2. ComfyUIの設定

ComfyUIで以下の設定を有効にする必要があります：

1. ComfyUIを起動
2. 設定 → "Enable Dev mode Options" をチェック
3. API経由でのアクセスが可能になります

### 3. ワークフローファイルの確認

以下のファイルが存在することを確認してください：
```
/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow.json
```

## 使用方法

### 1. APIサーバーの起動

```bash
python api_server.py
```

サーバーは `http://localhost:8000` で起動します。

### 2. API仕様の確認

ブラウザで以下にアクセスしてSwagger UIを確認：
```
http://localhost:8000/docs
```

### 3. 基本的な使用例

```python
import requests

# 画像をアップロードして3Dキャラクター生成
with open('input_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'prompt': 'anime character with detailed armor',
        'image_size': 768,
        'steps': 15
    }
    
    response = requests.post('http://localhost:8000/generate_character', 
                           files=files, data=data)
    
    job = response.json()
    job_id = job['job_id']

# ジョブの進捗を確認
status_response = requests.get(f'http://localhost:8000/job/{job_id}/status')
status = status_response.json()
print(f"Status: {status['status']}, Progress: {status['progress']}%")
```

### 4. 詳細な例

```bash
python example_usage.py
```

## API エンドポイント

### POST /generate_character

3Dキャラクター生成を開始

**パラメータ:**

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|-----------|------|------|-----------|------|
| `image` | File | Yes | - | 入力画像ファイル |
| `prompt` | String | No | "Highly Detailed" | テクスチャ生成用プロンプト |
| `negative_prompt` | String | No | "watermark, ugly..." | ネガティブプロンプト |
| `image_size` | Integer | No | 768 | マルチビュー生成サイズ |
| `steps` | Integer | No | 10 | 生成ステップ数 |
| `guidance_scale` | Float | No | 5.5 | ガイダンススケール |
| `seed` | Integer | No | -1 | ランダムシード |
| `enable_rigging` | Boolean | No | true | リギング有効化 |

**レスポンス:**
```json
{
  "status": "accepted",
  "job_id": "uuid-string",
  "message": "Character generation started",
  "estimated_time": "120-300 seconds"
}
```

### GET /job/{job_id}/status

ジョブの進捗状況を取得

**レスポンス:**
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
  ]
}
```

### GET /models

利用可能なモデル一覧を取得

### GET /health

APIの健康状態を確認

## ファイル構成

```
workflow_api/
├── api_server.py           # FastAPI サーバー
├── comfyui_client.py       # ComfyUI クライアント
├── requirements.txt        # Python依存関係
├── example_usage.py        # 使用例
├── workflow_analysis.md    # ワークフロー分析
├── api_spec.md            # API仕様
└── README.md              # このファイル
```

## 開発者向け情報

### ワークフローのカスタマイズ

`comfyui_client.py` の `WorkflowManager` クラスを編集して、ワークフローのパラメータを調整できます。

### 新しいノードの追加

1. `workflow_analysis.md` でノードIDを確認
2. `WorkflowManager` クラスに新しいメソッドを追加
3. API仕様を更新

### エラーハンドリング

APIは以下のエラーコードを返します：

- `400`: 無効なパラメータ
- `404`: ジョブが見つからない  
- `413`: ファイルサイズ制限超過
- `500`: サーバー内部エラー

## トラブルシューティング

### ComfyUIに接続できない

1. ComfyUIが起動していることを確認
2. `COMFYUI_SERVER` の設定を確認
3. ファイアウォール設定を確認

### ワークフロー実行エラー

1. 必要なモデルファイルが存在することを確認
2. ComfyUIの設定で開発者モードが有効化されていることを確認
3. GPUメモリが十分であることを確認

### アップロードエラー

1. ファイルサイズが10MB以下であることを確認
2. サポートされている画像形式 (JPG, PNG, WebP) を使用
3. ファイル権限を確認

## パフォーマンス最適化

- `steps` を下げると速度向上（品質低下）
- `image_size` を下げるとメモリ使用量削減
- `enable_rigging=false` でリギングをスキップして高速化
- 複数のジョブを並列実行する際はGPUメモリに注意

## ライセンス

このプロジェクトは研究・開発目的で作成されています。