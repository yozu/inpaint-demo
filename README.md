# Image Inpainting Web App

背景パターンを保持しながらオブジェクトを消去する Image Inpainting Webアプリケーション。

## 機能

複数の Inpainting 手法をサポート:

| 手法 | 説明 | 必要な環境 |
|------|------|-----------|
| OpenCV NS | Navier-Stokes法 | 標準 |
| OpenCV Telea | Fast Marching法 | 標準 |
| Texture Aware | テクスチャ考慮型（独自実装） | 標準 |
| LaMa | AI (Large Mask Inpainting) | `simple-lama-inpainting` |
| Stable Diffusion | AI (Diffusion Model) | `diffusers` + GPU |
| ClipDrop | Cloud API | APIキー |
| Replicate | Cloud API | APIキー |

## セットアップ

### 基本インストール

```bash
# Python 3.12 推奨
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### LaMa (ローカルAI) を使用する場合

```bash
pip install simple-lama-inpainting
```

初回実行時にモデル（約200MB）が自動ダウンロードされます。

## 起動

```bash
# Windows
start.bat

# または直接
python app.py
```

ブラウザで http://localhost:5001 を開いてください。

## 使い方

1. 画像をアップロード（またはデモ画像を使用）
2. 消したい部分を赤色でマスク
3. Inpainting 手法を選択
4. 「実行」ボタンをクリック

## スクリーンショット

（準備中）

## ライセンス

MIT
