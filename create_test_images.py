"""
テスト画像生成スクリプト

質問にあった例（斜線背景 + 棒人間）を再現する
"""

import cv2
import numpy as np


def create_striped_background(width: int, height: int, stripe_width: int = 20) -> np.ndarray:
    """斜線パターンの背景を生成"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白背景

    # 斜線を描画
    for i in range(-height, width + height, stripe_width * 2):
        pt1 = (i, 0)
        pt2 = (i + height, height)
        cv2.line(image, pt1, pt2, (0, 0, 0), 2)  # 黒線

    return image


def create_grid_background(width: int, height: int, grid_size: int = 30) -> np.ndarray:
    """格子パターンの背景を生成"""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 縦線
    for x in range(0, width, grid_size):
        cv2.line(image, (x, 0), (x, height), (200, 200, 200), 1)

    # 横線
    for y in range(0, height, grid_size):
        cv2.line(image, (0, y), (width, y), (200, 200, 200), 1)

    return image


def draw_stick_figure(image: np.ndarray, cx: int, cy: int, scale: float = 1.0) -> np.ndarray:
    """棒人間を描画"""
    result = image.copy()

    # 頭
    head_radius = int(20 * scale)
    cv2.circle(result, (cx, cy - int(60 * scale)), head_radius, (0, 0, 0), 2)

    # 体
    body_top = (cx, cy - int(40 * scale))
    body_bottom = (cx, cy + int(20 * scale))
    cv2.line(result, body_top, body_bottom, (0, 0, 0), 2)

    # 腕
    arm_y = cy - int(30 * scale)
    cv2.line(result, (cx, arm_y), (cx - int(30 * scale), arm_y + int(20 * scale)), (0, 0, 0), 2)
    cv2.line(result, (cx, arm_y), (cx + int(30 * scale), arm_y + int(20 * scale)), (0, 0, 0), 2)

    # 足
    cv2.line(result, body_bottom, (cx - int(20 * scale), cy + int(60 * scale)), (0, 0, 0), 2)
    cv2.line(result, body_bottom, (cx + int(20 * scale), cy + int(60 * scale)), (0, 0, 0), 2)

    return result


def create_mask_for_stick_figure(
    width: int,
    height: int,
    cx: int,
    cy: int,
    scale: float = 1.0,
    padding: int = 10
) -> np.ndarray:
    """棒人間部分のマスクを生成"""
    mask = np.zeros((height, width), dtype=np.uint8)

    # 棒人間を白で描画（太めに）
    # 頭
    head_radius = int(20 * scale) + padding
    cv2.circle(mask, (cx, cy - int(60 * scale)), head_radius, 255, -1)

    # 体（太い線）
    body_top = (cx, cy - int(40 * scale))
    body_bottom = (cx, cy + int(20 * scale))
    cv2.line(mask, body_top, body_bottom, 255, padding * 2)

    # 腕
    arm_y = cy - int(30 * scale)
    cv2.line(mask, (cx, arm_y), (cx - int(30 * scale), arm_y + int(20 * scale)), 255, padding * 2)
    cv2.line(mask, (cx, arm_y), (cx + int(30 * scale), arm_y + int(20 * scale)), 255, padding * 2)

    # 足
    cv2.line(mask, body_bottom, (cx - int(20 * scale), cy + int(60 * scale)), 255, padding * 2)
    cv2.line(mask, body_bottom, (cx + int(20 * scale), cy + int(60 * scale)), 255, padding * 2)

    return mask


def main():
    width, height = 400, 400

    # --- テスト1: 斜線背景 + 棒人間 ---
    print("テスト画像1: 斜線背景 + 棒人間")
    striped_bg = create_striped_background(width, height)
    test1_image = draw_stick_figure(striped_bg, 200, 200)
    test1_mask = create_mask_for_stick_figure(width, height, 200, 200)

    cv2.imwrite("test_striped_input.png", test1_image)
    cv2.imwrite("test_striped_mask.png", test1_mask)
    print("  - test_striped_input.png")
    print("  - test_striped_mask.png")

    # --- テスト2: 格子背景 + 棒人間 ---
    print("\nテスト画像2: 格子背景 + 棒人間")
    grid_bg = create_grid_background(width, height)
    test2_image = draw_stick_figure(grid_bg, 200, 200)
    test2_mask = create_mask_for_stick_figure(width, height, 200, 200)

    cv2.imwrite("test_grid_input.png", test2_image)
    cv2.imwrite("test_grid_mask.png", test2_mask)
    print("  - test_grid_input.png")
    print("  - test_grid_mask.png")

    # --- テスト3: 単色背景 + 棒人間（比較用） ---
    print("\nテスト画像3: 単色背景 + 棒人間（比較用）")
    solid_bg = np.ones((height, width, 3), dtype=np.uint8) * 240
    test3_image = draw_stick_figure(solid_bg, 200, 200)
    test3_mask = create_mask_for_stick_figure(width, height, 200, 200)

    cv2.imwrite("test_solid_input.png", test3_image)
    cv2.imwrite("test_solid_mask.png", test3_mask)
    print("  - test_solid_input.png")
    print("  - test_solid_mask.png")

    print("\nテスト画像の生成が完了しました。")


if __name__ == "__main__":
    main()
