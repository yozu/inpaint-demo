"""
AIモデルを使用したInpainting実装

利用可能なモデル:
1. LaMa (Large Mask Inpainting) - 軽量で高品質
2. Stable Diffusion Inpainting - 最高品質だが重い
"""

import cv2
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LamaInpainter:
    """
    LaMa (Large Mask Inpainting) を使用したInpainting

    特徴:
    - Fourier Convolutionを使用した高品質な修復
    - 大きなマスク領域でも自然な結果
    - 比較的軽量（GPU推奨だがCPUでも動作）
    """

    def __init__(self):
        self.model = None
        self.device = None

    def _ensure_model_loaded(self):
        """モデルの遅延ロード"""
        if self.model is not None:
            return

        try:
            from simple_lama_inpainting import SimpleLama
            self.model = SimpleLama()
            print("LaMaモデルをロードしました")
        except ImportError:
            raise ImportError(
                "simple-lama-inpainting がインストールされていません。\n"
                "インストール: pip install simple-lama-inpainting"
            )

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        LaMaでInpainting実行

        Args:
            image: 入力画像 (H, W, 3) BGR形式
            mask: マスク画像 (H, W) 255=削除対象

        Returns:
            修復後の画像
        """
        self._ensure_model_loaded()

        from PIL import Image

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NumPy -> PIL
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(mask)

        # Inpainting実行
        result = self.model(pil_image, pil_mask)

        # PIL -> NumPy -> BGR
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr


class StableDiffusionInpainter:
    """
    Stable Diffusion Inpainting Pipeline

    特徴:
    - 最高品質の結果
    - テキストプロンプトで制御可能
    - GPU必須、VRAM 4GB以上推奨
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting"):
        self.model_id = model_id
        self.pipe = None
        self.device = None

    def _ensure_model_loaded(self):
        """モデルの遅延ロード"""
        if self.pipe is not None:
            return

        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            # デバイス選択
            if torch.cuda.is_available():
                self.device = "cuda"
                torch_dtype = torch.float16
            else:
                self.device = "cpu"
                torch_dtype = torch.float32
                print("警告: GPUが利用できないためCPUで実行します（非常に遅い）")

            print(f"Stable Diffusionモデルをロード中... ({self.device})")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
            )
            self.pipe = self.pipe.to(self.device)

            # メモリ最適化
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()

            print("Stable Diffusionモデルをロードしました")

        except ImportError:
            raise ImportError(
                "diffusers がインストールされていません。\n"
                "インストール: pip install diffusers transformers accelerate torch"
            )

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "background pattern, seamless texture",
        negative_prompt: str = "person, human, figure, blurry",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> np.ndarray:
        """
        Stable DiffusionでInpainting実行

        Args:
            image: 入力画像 (H, W, 3) BGR形式
            mask: マスク画像 (H, W) 255=削除対象
            prompt: 生成プロンプト
            negative_prompt: ネガティブプロンプト
            num_inference_steps: 推論ステップ数
            guidance_scale: ガイダンススケール

        Returns:
            修復後の画像
        """
        self._ensure_model_loaded()

        from PIL import Image

        # 画像サイズを512x512にリサイズ（SD要件）
        original_size = (image.shape[1], image.shape[0])
        target_size = (512, 512)

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # リサイズ
        image_resized = cv2.resize(image_rgb, target_size)
        mask_resized = cv2.resize(mask, target_size)

        # NumPy -> PIL
        pil_image = Image.fromarray(image_resized)
        pil_mask = Image.fromarray(mask_resized)

        # Inpainting実行
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # PIL -> NumPy
        result_np = np.array(result)

        # 元のサイズに戻す
        result_resized = cv2.resize(result_np, original_size)

        # RGB -> BGR
        result_bgr = cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR)

        return result_bgr


class IOPaintInpainter:
    """
    IOPaint (旧lama-cleaner) を使用したInpainting

    複数のモデルをサポート:
    - lama: LaMa
    - ldm: Latent Diffusion Model
    - zits: ZITS
    - mat: MAT
    - fcf: FcF
    - manga: Manga Inpainting
    """

    def __init__(self, model_name: str = "lama"):
        self.model_name = model_name
        self.model = None

    def _ensure_model_loaded(self):
        if self.model is not None:
            return

        try:
            from iopaint import create_model
            from iopaint.model_manager import ModelManager

            print(f"IOPaint ({self.model_name}) モデルをロード中...")
            self.model = create_model(self.model_name)
            print(f"IOPaint ({self.model_name}) モデルをロードしました")

        except ImportError:
            raise ImportError(
                "iopaint がインストールされていません。\n"
                "インストール: pip install iopaint"
            )

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """IOPaintでInpainting実行"""
        self._ensure_model_loaded()

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inpainting実行
        result = self.model(image_rgb, mask)

        # RGB -> BGR
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return result_bgr


# --- 簡易インターフェース ---

def ai_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "lama"
) -> np.ndarray:
    """
    AIモデルでInpainting実行

    Args:
        image: 入力画像 (BGR)
        mask: マスク (255=削除対象)
        method: "lama", "sd", "iopaint"

    Returns:
        修復後の画像
    """
    if method == "lama":
        inpainter = LamaInpainter()
    elif method == "sd":
        inpainter = StableDiffusionInpainter()
    elif method == "iopaint":
        inpainter = IOPaintInpainter()
    else:
        raise ValueError(f"Unknown method: {method}")

    return inpainter.inpaint(image, mask)


def check_available_models() -> dict:
    """利用可能なモデルをチェック"""
    available = {}

    # LaMa
    try:
        import simple_lama_inpainting
        available['lama'] = True
    except ImportError:
        available['lama'] = False

    # Stable Diffusion
    try:
        import diffusers
        import torch
        available['sd'] = True
        available['sd_gpu'] = torch.cuda.is_available()
    except ImportError:
        available['sd'] = False
        available['sd_gpu'] = False

    # IOPaint
    try:
        import iopaint
        available['iopaint'] = True
    except ImportError:
        available['iopaint'] = False

    return available


if __name__ == "__main__":
    print("AIモデル利用可能状況:")
    print("-" * 40)

    status = check_available_models()

    for model, available in status.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {model}")

    print("-" * 40)
    print("\nインストール方法:")
    print("  LaMa:              pip install simple-lama-inpainting")
    print("  Stable Diffusion:  pip install diffusers transformers accelerate torch")
    print("  IOPaint:           pip install iopaint")
