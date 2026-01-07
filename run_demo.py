"""
デモ実行スクリプト

テスト画像を生成し、各手法で比較実行する
"""

import cv2
import numpy as np
import os

# 同じディレクトリの他のスクリプトをインポート
from create_test_images import main as create_test_images
from texture_aware_inpaint import (
    TextureAwareInpainter,
    opencv_inpaint_comparison
)


def run_comparison(
    input_path: str,
    mask_path: str,
    output_prefix: str
):
    """各手法で比較実行"""
    print(f"\n処理中: {input_path}")

    image = cv2.imread(input_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"  エラー: 画像を読み込めません")
        return

    # 1. OpenCV INPAINT_NS
    result_ns = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    cv2.imwrite(f"{output_prefix}_opencv_ns.png", result_ns)
    print(f"  - {output_prefix}_opencv_ns.png (OpenCV Navier-Stokes)")

    # 2. OpenCV INPAINT_TELEA
    result_telea = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(f"{output_prefix}_opencv_telea.png", result_telea)
    print(f"  - {output_prefix}_opencv_telea.png (OpenCV Telea)")

    # 3. テクスチャ対応Inpainting
    inpainter = TextureAwareInpainter(patch_size=9, search_area=80)
    result_texture = inpainter.inpaint(image, mask)
    cv2.imwrite(f"{output_prefix}_texture_aware.png", result_texture)
    print(f"  - {output_prefix}_texture_aware.png (テクスチャ対応)")


def create_comparison_image(
    input_path: str,
    mask_path: str,
    result_paths: list,
    output_path: str,
    labels: list
):
    """比較画像を1枚にまとめる"""
    images = []

    # 入力画像
    img = cv2.imread(input_path)
    cv2.putText(img, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    images.append(img)

    # マスク
    mask = cv2.imread(mask_path)
    cv2.putText(mask, "Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    images.append(mask)

    # 結果
    for path, label in zip(result_paths, labels):
        if os.path.exists(path):
            result = cv2.imread(path)
            cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            images.append(result)

    # 横に並べる
    combined = np.hstack(images)
    cv2.imwrite(output_path, combined)
    print(f"比較画像: {output_path}")


def main():
    print("=" * 60)
    print("テクスチャ対応 Image Inpainting デモ")
    print("=" * 60)

    # 1. テスト画像生成
    print("\n[Step 1] テスト画像を生成中...")
    create_test_images()

    # 2. 各テストケースで比較実行
    print("\n[Step 2] Inpainting処理を実行中...")

    test_cases = [
        ("test_striped_input.png", "test_striped_mask.png", "result_striped"),
        ("test_grid_input.png", "test_grid_mask.png", "result_grid"),
        ("test_solid_input.png", "test_solid_mask.png", "result_solid"),
    ]

    for input_path, mask_path, output_prefix in test_cases:
        if os.path.exists(input_path) and os.path.exists(mask_path):
            run_comparison(input_path, mask_path, output_prefix)

    # 3. 比較画像を作成
    print("\n[Step 3] 比較画像を作成中...")

    for input_path, mask_path, output_prefix in test_cases:
        result_paths = [
            f"{output_prefix}_opencv_ns.png",
            f"{output_prefix}_opencv_telea.png",
            f"{output_prefix}_texture_aware.png",
        ]
        labels = ["OpenCV NS", "OpenCV Telea", "Texture-Aware"]

        if all(os.path.exists(p) for p in result_paths):
            create_comparison_image(
                input_path, mask_path,
                result_paths, f"{output_prefix}_comparison.png",
                labels
            )

    print("\n" + "=" * 60)
    print("完了しました！")
    print("\n結果:")
    print("  - result_striped_comparison.png : 斜線背景の比較")
    print("  - result_grid_comparison.png    : 格子背景の比較")
    print("  - result_solid_comparison.png   : 単色背景の比較")
    print("")
    print("ポイント:")
    print("  - OpenCV標準: 単色背景では有効だがパターン背景は消える")
    print("  - テクスチャ対応: パターンを検出して維持しながら修復")
    print("=" * 60)


if __name__ == "__main__":
    main()
