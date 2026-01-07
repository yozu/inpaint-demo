# 背景パターンを維持しながら対象物を消去する方法

## 質問への回答

### 問題の本質

Canvaの「マジック消しゴム」のような機能で、**背景に模様がある場合、対象物と一緒に模様まで消えてしまう**問題。

---

## 🔑 結論

| アプローチ | 品質 | 容量 | 費用 |
|-----------|------|------|------|
| OpenCV標準 | ❌ 模様消失 | 0MB | 無料 |
| PatchMatch/テクスチャ合成 | △ 改善するが限界あり | 0MB | 無料 |
| **LaMa (AI)** | ✅ 高品質 | 約200MB | 無料 |
| **Stable Diffusion** | ✅ 最高品質 | 約4GB | 無料 |
| **クラウドAPI** | ✅ 高品質 | **0MB** | 従量課金 |

**推奨**:
- 容量気にしない → **LaMa** (ローカル)
- 容量節約 → **クラウドAPI** (Replicate, ClipDrop)

---

## 方法1: 従来手法（AIなし）

### OpenCV標準 `cv2.inpaint()`

```python
import cv2

# Navier-Stokes方程式ベース
result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

# Fast Marching Method
result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
```

**問題**: 拡散ベースなので、周囲の色を平均化してパターンが消える

### PatchMatch + テクスチャ合成

```python
from texture_aware_inpaint import TextureAwareInpainter

inpainter = TextureAwareInpainter(patch_size=9, search_area=50)
result = inpainter.inpaint(image, mask)
```

**改善点**: 画像内から類似パッチを検索してコピー
**限界**: 完璧ではない、輪郭が残ることがある

---

## 方法2: AIモデル（ローカル）

### LaMa (推奨)

```bash
pip install simple-lama-inpainting
```

```python
from simple_lama_inpainting import SimpleLama
from PIL import Image

lama = SimpleLama()
result = lama(pil_image, pil_mask)
```

**特徴**:
- Fourier Convolutionで高品質な修復
- 大きなマスクでも自然な結果
- 約200MBのモデル

### Stable Diffusion Inpainting

```bash
pip install diffusers transformers accelerate torch
```

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
)
result = pipe(prompt="background", image=image, mask_image=mask)
```

**特徴**:
- 最高品質
- プロンプトで制御可能
- 約4GBのモデル、GPU推奨

---

## 方法3: クラウドAPI（容量節約）

### Replicate API

```python
import replicate

output = replicate.run(
    "andreasjansson/lama:b37a2ff...",
    input={"image": img_uri, "mask": mask_uri}
)
```

**料金**: 約$0.0002/回

### ClipDrop API

```python
import requests

response = requests.post(
    'https://clipdrop-api.co/cleanup/v1',
    files={'image_file': img, 'mask_file': mask},
    headers={'x-api-key': API_KEY}
)
```

**料金**: 月100回無料、以降$0.01/回

### Hugging Face Inference API

```python
import requests

response = requests.post(
    "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    files={"image": img, "mask": mask}
)
```

**料金**: 無料枠あり（レート制限付き）

---

## デモアプリ

### 起動方法

```bash
cd inpaint_demo
pip install -r requirements.txt
python app.py
```

ブラウザで http://localhost:5000 を開く

### ファイル構成

```
inpaint_demo/
├── app.py                    # Webサーバー
├── texture_aware_inpaint.py  # 従来手法
├── ai_inpaint.py             # AIモデル（ローカル）
├── cloud_inpaint.py          # クラウドAPI
├── templates/index.html      # Web UI
└── requirements.txt
```

---

## まとめ

### なぜ従来手法では難しいか

1. **拡散ベース**（OpenCV）: 周囲の色を平均化 → パターン消失
2. **パッチベース**（PatchMatch）: 類似パッチをコピー → 境界に違和感

### なぜAIモデルは上手くいくか

1. **コンテキスト理解**: 周囲の構造を「理解」して自然に補完
2. **大量の学習データ**: 様々なパターンの修復方法を学習済み
3. **生成能力**: 存在しない部分を「生成」できる

### 質問への最終回答

> 「背景のパターンや線を残しつつ、対象物だけを消去する方法」

**外部AIモデルなし**: PatchMatch/テクスチャ合成である程度改善できるが、完璧ではない

**AIモデル使用**: LaMa または Stable Diffusion で高品質な結果が得られる

**容量を節約したい**: Replicate, ClipDrop 等のクラウドAPIを使用
