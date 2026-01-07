"""
クラウドAPIを使用したInpainting

ローカルにモデルをダウンロードせずに
AIの品質を試せる

利用可能なAPI:
1. Replicate API (LaMa, Stable Diffusion)
2. Hugging Face Inference API
3. ClipDrop API
"""

import cv2
import numpy as np
import base64
import requests
import os
from typing import Optional
import time


class ReplicateInpainter:
    """
    Replicate API を使用したInpainting

    無料枠: 新規登録で少し使える
    料金: 従量課金（安い）

    https://replicate.com/
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError(
                "Replicate APIトークンが必要です。\n"
                "1. https://replicate.com/ でアカウント作成\n"
                "2. APIトークンを取得\n"
                "3. 環境変数 REPLICATE_API_TOKEN に設定"
            )

    def inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMaモデルでInpainting"""
        import replicate

        # 画像をBase64エンコード
        _, img_buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        _, mask_buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        mask_data_uri = f"data:image/png;base64,{mask_base64}"

        # LaMaモデルを実行
        output = replicate.run(
            "andreasjansson/lama:b37a2ffdd12ba3f64e12c2ed6b444a6f6c46d51a0519dc4db2a8e1ce0f6aa5c5",
            input={
                "image": img_data_uri,
                "mask": mask_data_uri
            }
        )

        # 結果をダウンロード
        response = requests.get(output)
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        result = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return result

    def inpaint_sdxl(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "background pattern, seamless"
    ) -> np.ndarray:
        """Stable Diffusion XL InpaintingでInpainting"""
        import replicate

        # 画像をBase64エンコード
        _, img_buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        _, mask_buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
        mask_data_uri = f"data:image/png;base64,{mask_base64}"

        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "image": img_data_uri,
                "mask": mask_data_uri,
                "prompt": prompt,
                "num_inference_steps": 25
            }
        )

        # 結果をダウンロード
        response = requests.get(output[0])
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        result = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return result


class HuggingFaceInpainter:
    """
    Hugging Face Inference API を使用したInpainting

    無料枠: あり（レート制限付き）
    https://huggingface.co/inference-api
    """

    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.environ.get('HF_API_TOKEN')
        if not self.api_token:
            raise ValueError(
                "Hugging Face APIトークンが必要です。\n"
                "1. https://huggingface.co/ でアカウント作成\n"
                "2. Settings > Access Tokens でトークン作成\n"
                "3. 環境変数 HF_API_TOKEN に設定"
            )
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        model: str = "runwayml/stable-diffusion-inpainting"
    ) -> np.ndarray:
        """Hugging Face APIでInpainting"""
        import io
        from PIL import Image

        # OpenCV -> PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(mask)

        # PIL -> bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        mask_bytes = io.BytesIO()
        pil_mask.save(mask_bytes, format='PNG')
        mask_bytes.seek(0)

        # API呼び出し
        api_url = f"https://api-inference.huggingface.co/models/{model}"

        files = {
            'image': img_bytes,
            'mask': mask_bytes
        }

        response = requests.post(
            api_url,
            headers=self.headers,
            files=files
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.text}")

        # 結果をデコード
        result_pil = Image.open(io.BytesIO(response.content))
        result_np = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr


class ClipDropInpainter:
    """
    ClipDrop API を使用したInpainting

    無料枠: 月100回
    https://clipdrop.co/apis
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('CLIPDROP_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ClipDrop APIキーが必要です。\n"
                "1. https://clipdrop.co/apis でアカウント作成\n"
                "2. APIキーを取得\n"
                "3. 環境変数 CLIPDROP_API_KEY に設定"
            )

    def cleanup(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ClipDrop Cleanup APIで不要なオブジェクトを削除"""
        # 画像をエンコード
        _, img_buffer = cv2.imencode('.png', image)
        _, mask_buffer = cv2.imencode('.png', mask)

        response = requests.post(
            'https://clipdrop-api.co/cleanup/v1',
            files={
                'image_file': ('image.png', img_buffer.tobytes(), 'image/png'),
                'mask_file': ('mask.png', mask_buffer.tobytes(), 'image/png'),
            },
            headers={'x-api-key': self.api_key}
        )

        if response.status_code != 200:
            raise Exception(f"ClipDrop API Error: {response.text}")

        # 結果をデコード
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        result = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return result


def check_cloud_apis() -> dict:
    """利用可能なクラウドAPIをチェック"""
    available = {}

    # Replicate
    available['replicate'] = bool(os.environ.get('REPLICATE_API_TOKEN'))

    # Hugging Face
    available['huggingface'] = bool(os.environ.get('HF_API_TOKEN'))

    # ClipDrop
    available['clipdrop'] = bool(os.environ.get('CLIPDROP_API_KEY'))

    return available


if __name__ == "__main__":
    print("クラウドAPI利用可能状況:")
    print("-" * 50)

    status = check_cloud_apis()

    for api, available in status.items():
        icon = "✅" if available else "❌"
        env_var = {
            'replicate': 'REPLICATE_API_TOKEN',
            'huggingface': 'HF_API_TOKEN',
            'clipdrop': 'CLIPDROP_API_KEY'
        }[api]
        print(f"  {icon} {api:15} (環境変数: {env_var})")

    print("-" * 50)
    print("\n設定方法 (Windows):")
    print('  set REPLICATE_API_TOKEN=r8_xxxxx')
    print('  set HF_API_TOKEN=hf_xxxxx')
    print('  set CLIPDROP_API_KEY=xxxxx')
    print("\n設定方法 (Linux/Mac):")
    print('  export REPLICATE_API_TOKEN=r8_xxxxx')
    print("\nAPI取得先:")
    print("  Replicate:   https://replicate.com/")
    print("  HuggingFace: https://huggingface.co/settings/tokens")
    print("  ClipDrop:    https://clipdrop.co/apis")
