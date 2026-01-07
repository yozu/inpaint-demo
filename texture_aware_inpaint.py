"""
テクスチャ対応のImage Inpainting実装

背景にパターンや模様がある場合でも、それを維持しながら
対象物を消去するアルゴリズム。

外部AIモデル不使用 - OpenCVとNumPyのみで実装
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class TextureAwareInpainter:
    """
    テクスチャを考慮したInpainting処理クラス

    従来のOpenCV inpaintは拡散ベースで単色背景向き。
    このクラスはPatchMatchとテクスチャ合成を組み合わせ、
    パターン背景でも自然な修復を実現する。
    """

    def __init__(self, patch_size: int = 9, search_area: int = 50):
        """
        Args:
            patch_size: パッチサイズ（奇数推奨）
            search_area: 類似パッチ検索範囲
        """
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.search_area = search_area

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        メインのInpainting処理

        Args:
            image: 入力画像 (H, W, 3) BGR形式
            mask: マスク画像 (H, W) 255=削除対象, 0=保持

        Returns:
            修復後の画像
        """
        # マスクを二値化
        mask = (mask > 127).astype(np.uint8) * 255

        # Step 1: 背景パターンの解析
        pattern_info = self._analyze_background_pattern(image, mask)

        # Step 2: パターンの種類に応じた処理を選択
        if pattern_info['has_regular_pattern']:
            # 規則的なパターン（ストライプ、格子など）
            result = self._inpaint_with_pattern_synthesis(image, mask, pattern_info)
        else:
            # 不規則なテクスチャまたは複雑な背景
            result = self._inpaint_with_patchmatch(image, mask)

        # Step 3: 境界をスムージング
        result = self._smooth_boundary(result, image, mask)

        return result

    def _analyze_background_pattern(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        背景のパターンを解析

        FFTを使用してパターンの周期性を検出
        """
        # マスク外の領域（背景）を取得
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg_mask = cv2.bitwise_not(mask)

        # マスクを拡張して削除対象周辺を除外
        kernel = np.ones((21, 21), np.uint8)
        bg_mask_eroded = cv2.erode(bg_mask, kernel, iterations=1)

        if cv2.countNonZero(bg_mask_eroded) < 100:
            return {'has_regular_pattern': False, 'period': None, 'angle': None}

        # 背景領域のサンプルを取得
        coords = np.where(bg_mask_eroded > 0)
        if len(coords[0]) == 0:
            return {'has_regular_pattern': False, 'period': None, 'angle': None}

        # サンプル領域からFFTでパターン検出
        sample_size = 64
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))

        y_start = max(0, center_y - sample_size // 2)
        y_end = min(image.shape[0], center_y + sample_size // 2)
        x_start = max(0, center_x - sample_size // 2)
        x_end = min(image.shape[1], center_x + sample_size // 2)

        sample = gray[y_start:y_end, x_start:x_end].astype(np.float32)

        if sample.size == 0 or sample.shape[0] < 16 or sample.shape[1] < 16:
            return {'has_regular_pattern': False, 'period': None, 'angle': None}

        # FFT解析
        f_transform = np.fft.fft2(sample)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # DC成分を除去
        center = (magnitude.shape[0] // 2, magnitude.shape[1] // 2)
        magnitude[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 0

        # 強いピークがあれば規則的パターン
        threshold = np.mean(magnitude) + 3 * np.std(magnitude)
        peaks = np.where(magnitude > threshold)

        has_pattern = len(peaks[0]) > 2

        period = None
        angle = None
        if has_pattern and len(peaks[0]) > 0:
            # 最も強いピークから周期と角度を推定
            max_idx = np.argmax(magnitude[peaks])
            peak_y, peak_x = peaks[0][max_idx], peaks[1][max_idx]

            dy = peak_y - center[0]
            dx = peak_x - center[1]

            if dy != 0 or dx != 0:
                period = sample.shape[0] / np.sqrt(dy**2 + dx**2)
                angle = np.arctan2(dy, dx) * 180 / np.pi

        return {
            'has_regular_pattern': has_pattern,
            'period': period,
            'angle': angle,
            'sample_region': (y_start, y_end, x_start, x_end)
        }

    def _inpaint_with_pattern_synthesis(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pattern_info: dict
    ) -> np.ndarray:
        """
        規則的パターンの合成による修復

        ストライプや格子パターンを検出し、
        パターンの周期性を維持しながら修復
        """
        result = image.copy()

        # 1. 背景からパターンタイルを抽出
        tile = self._extract_pattern_tile(image, mask, pattern_info)

        if tile is None:
            # タイル抽出失敗時はPatchMatchにフォールバック
            return self._inpaint_with_patchmatch(image, mask)

        # 2. タイルを使用してマスク領域を埋める
        coords = np.where(mask > 0)
        for y, x in zip(coords[0], coords[1]):
            # タイル内の対応位置を計算
            tile_y = y % tile.shape[0]
            tile_x = x % tile.shape[1]
            result[y, x] = tile[tile_y, tile_x]

        # 3. 境界の不連続性を修正
        result = self._blend_boundaries(result, image, mask)

        return result

    def _extract_pattern_tile(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pattern_info: dict
    ) -> Optional[np.ndarray]:
        """
        繰り返しパターンのタイルを抽出
        """
        period = pattern_info.get('period')

        if period is None or period < 4:
            period = 20  # デフォルトのタイルサイズ

        tile_size = int(period * 2)  # 2周期分を確保
        tile_size = max(8, min(tile_size, 100))  # 8〜100の範囲に制限

        # マスク外で最も均質な領域を探す
        bg_mask = cv2.bitwise_not(mask)
        kernel = np.ones((tile_size, tile_size), np.uint8)

        # タイルが完全にマスク外に収まる位置を探す
        valid_positions = cv2.erode(bg_mask, kernel, iterations=1)
        coords = np.where(valid_positions > 0)

        if len(coords[0]) == 0:
            return None

        # 複数候補からばらつきの少ないものを選択
        best_tile = None
        min_variance = float('inf')

        for _ in range(min(10, len(coords[0]))):
            idx = np.random.randint(len(coords[0]))
            y, x = coords[0][idx], coords[1][idx]

            if y + tile_size > image.shape[0] or x + tile_size > image.shape[1]:
                continue

            candidate = image[y:y+tile_size, x:x+tile_size].copy()
            variance = np.var(candidate)

            # 完全に単色でなければ採用
            if 10 < variance < min_variance:
                min_variance = variance
                best_tile = candidate

        return best_tile

    def _inpaint_with_patchmatch(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        PatchMatchアルゴリズムによるInpainting

        マスク領域の各ピクセルに対して、
        マスク外から最も類似するパッチを検索して合成
        """
        result = image.copy()
        mask_binary = (mask > 0).astype(np.uint8)

        # 修復順序を決定（境界から内側へ）
        fill_order = self._compute_fill_order(mask_binary)

        # 信頼度マップ（既知領域=1, 未知=0）
        confidence = (1 - mask_binary).astype(np.float32)

        # グレースケール変換（パッチマッチング用）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 優先順位キュー（境界ピクセル）
        for y, x in fill_order:
            if mask_binary[y, x] == 0:
                continue

            # このピクセルの最適なソースパッチを検索
            best_patch_pos = self._find_best_patch(
                result, gray, mask_binary, confidence, y, x
            )

            if best_patch_pos is not None:
                # パッチをコピー
                self._copy_patch(
                    result, mask_binary, confidence,
                    y, x, best_patch_pos[0], best_patch_pos[1]
                )

        return result

    def _compute_fill_order(self, mask: np.ndarray) -> list:
        """
        境界から内側への修復順序を計算

        Onion-peel方式：外側から1ピクセルずつ修復
        """
        fill_order = []
        remaining = mask.copy()

        while cv2.countNonZero(remaining) > 0:
            # 現在の境界を取得
            boundary = self._get_boundary_pixels(remaining)
            fill_order.extend(boundary)

            # 境界ピクセルを処理済みにする
            for y, x in boundary:
                remaining[y, x] = 0

        return fill_order

    def _get_boundary_pixels(self, mask: np.ndarray) -> list:
        """マスク領域の境界ピクセルを取得"""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - eroded

        coords = np.where(boundary > 0)
        return list(zip(coords[0], coords[1]))

    def _find_best_patch(
        self,
        image: np.ndarray,
        gray: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        target_y: int,
        target_x: int
    ) -> Optional[Tuple[int, int]]:
        """
        ターゲット位置に最適なソースパッチを検索
        """
        h, w = image.shape[:2]
        half = self.half_patch

        # ターゲットパッチ（マスク外部分のみ使用）
        ty_start = max(0, target_y - half)
        ty_end = min(h, target_y + half + 1)
        tx_start = max(0, target_x - half)
        tx_end = min(w, target_x + half + 1)

        target_patch = gray[ty_start:ty_end, tx_start:tx_end]
        target_mask = mask[ty_start:ty_end, tx_start:tx_end]

        # 既知ピクセルが少なすぎる場合
        known_ratio = 1 - np.mean(target_mask)
        if known_ratio < 0.3:
            # 最近傍の既知ピクセルを使用
            return self._find_nearest_source(mask, target_y, target_x)

        best_pos = None
        best_score = float('inf')

        # 検索範囲を設定
        search_y_start = max(half, target_y - self.search_area)
        search_y_end = min(h - half - 1, target_y + self.search_area)
        search_x_start = max(half, target_x - self.search_area)
        search_x_end = min(w - half - 1, target_x + self.search_area)

        # グリッドサンプリングで効率化
        step = max(1, self.search_area // 20)

        for sy in range(search_y_start, search_y_end, step):
            for sx in range(search_x_start, search_x_end, step):
                # ソースパッチがマスク領域と重ならないか確認
                source_mask = mask[sy-half:sy+half+1, sx-half:sx+half+1]
                if source_mask.size == 0 or np.any(source_mask > 0):
                    continue

                # パッチの類似度を計算
                source_patch = gray[sy-half:sy+half+1, sx-half:sx+half+1]

                if source_patch.shape != target_patch.shape:
                    continue

                # 既知ピクセルのみで比較
                valid_mask = (target_mask == 0)
                if not np.any(valid_mask):
                    continue

                diff = np.abs(source_patch - target_patch)
                score = np.mean(diff[valid_mask])

                if score < best_score:
                    best_score = score
                    best_pos = (sy, sx)

        return best_pos

    def _find_nearest_source(
        self,
        mask: np.ndarray,
        y: int,
        x: int
    ) -> Optional[Tuple[int, int]]:
        """マスク外で最も近い位置を検索"""
        h, w = mask.shape

        for radius in range(1, max(h, w)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) != radius and abs(dx) != radius:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                        return (ny, nx)
        return None

    def _copy_patch(
        self,
        result: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
        target_y: int,
        target_x: int,
        source_y: int,
        source_x: int
    ):
        """ソースパッチをターゲット位置にコピー"""
        half = self.half_patch
        h, w = result.shape[:2]

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ty, tx = target_y + dy, target_x + dx
                sy, sx = source_y + dy, source_x + dx

                if not (0 <= ty < h and 0 <= tx < w):
                    continue
                if not (0 <= sy < h and 0 <= sx < w):
                    continue

                # マスク領域のみ更新
                if mask[ty, tx] > 0:
                    result[ty, tx] = result[sy, sx]
                    mask[ty, tx] = 0
                    confidence[ty, tx] = 1.0

    def _blend_boundaries(
        self,
        result: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """境界をブレンドして滑らかに"""
        # マスク境界をぼかす
        mask_float = mask.astype(np.float32) / 255.0
        blurred_mask = cv2.GaussianBlur(mask_float, (11, 11), 0)

        # 3チャンネルに拡張
        blurred_mask_3ch = np.stack([blurred_mask] * 3, axis=-1)

        # ブレンド
        blended = (result * blurred_mask_3ch +
                   original * (1 - blurred_mask_3ch))

        return blended.astype(np.uint8)

    def _smooth_boundary(
        self,
        result: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """最終的な境界スムージング"""
        # Poisson Blending風の処理（簡易版）
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        boundary_region = dilated - cv2.erode(mask, kernel, iterations=2)

        # 境界領域のみガウシアンブラー
        blurred = cv2.GaussianBlur(result, (5, 5), 0)

        boundary_mask = boundary_region > 0
        result[boundary_mask] = blurred[boundary_mask]

        return result


def remove_object_preserve_pattern(
    image_path: str,
    mask_path: str,
    output_path: str,
    patch_size: int = 9,
    search_area: int = 50
) -> np.ndarray:
    """
    対象物を削除しつつ背景パターンを維持

    Args:
        image_path: 入力画像パス
        mask_path: マスク画像パス（削除対象=白）
        output_path: 出力画像パス
        patch_size: パッチサイズ
        search_area: 検索範囲

    Returns:
        修復後の画像
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    if mask is None:
        raise ValueError(f"マスクを読み込めません: {mask_path}")

    # サイズを合わせる
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    inpainter = TextureAwareInpainter(patch_size, search_area)
    result = inpainter.inpaint(image, mask)

    cv2.imwrite(output_path, result)
    print(f"修復完了: {output_path}")

    return result


# --- 比較用: OpenCV標準のInpainting ---

def opencv_inpaint_comparison(
    image_path: str,
    mask_path: str,
    output_ns_path: str,
    output_telea_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV標準のInpainting（比較用）

    これらは単色背景では有効だが、
    パターン背景では背景が消えてしまう問題がある
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Navier-Stokes方程式ベース
    result_ns = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    cv2.imwrite(output_ns_path, result_ns)

    # Fast Marching Method
    result_telea = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_telea_path, result_telea)

    return result_ns, result_telea


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("使用方法: python texture_aware_inpaint.py <入力画像> <マスク画像> <出力画像>")
        print("")
        print("例: python texture_aware_inpaint.py input.png mask.png output.png")
        print("")
        print("マスク画像: 削除したい対象を白(255)で塗りつぶした画像")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_path = sys.argv[3]

    remove_object_preserve_pattern(image_path, mask_path, output_path)
