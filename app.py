"""
Webインターフェース - Flask版

ブラウザから画像をアップロードし、
マスク描画 → Inpainting実行 → 結果表示
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
from texture_aware_inpaint import TextureAwareInpainter

# AIモデル（オプション）
AI_MODELS_AVAILABLE = {}
try:
    from ai_inpaint import LamaInpainter, check_available_models
    AI_MODELS_AVAILABLE = check_available_models()
    print(f"[INIT] AI Models detected: {AI_MODELS_AVAILABLE}")
except ImportError as e:
    print(f"[INIT] AI import failed: {e}")

# クラウドAPI
CLOUD_APIS_AVAILABLE = {}
try:
    from cloud_inpaint import check_cloud_apis
    CLOUD_APIS_AVAILABLE = check_cloud_apis()
except ImportError:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB上限

# アップロードされた画像を一時保存
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def base64_to_cv2(base64_str: str) -> np.ndarray:
    """Base64文字列をOpenCV画像に変換"""
    # data:image/png;base64, のプレフィックスを除去
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def cv2_to_base64(img: np.ndarray) -> str:
    """OpenCV画像をBase64文字列に変換"""
    _, buffer = cv2.imencode('.png', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


# APIキーをメモリに保持（セッション中のみ有効）
api_keys = {
    'clipdrop': os.environ.get('CLIPDROP_API_KEY', ''),
    'replicate': os.environ.get('REPLICATE_API_TOKEN', '')
}


@app.route('/api-keys', methods=['GET'])
def get_api_keys():
    """保存されたAPIキーを取得（マスク表示）"""
    return jsonify({
        'clipdrop': '***' + api_keys['clipdrop'][-4:] if len(api_keys['clipdrop']) > 4 else '',
        'replicate': '***' + api_keys['replicate'][-4:] if len(api_keys['replicate']) > 4 else ''
    })


@app.route('/api-keys', methods=['POST'])
def save_api_keys():
    """APIキーを保存"""
    global api_keys, CLOUD_APIS_AVAILABLE

    data = request.json

    if data.get('clipdrop') and not data['clipdrop'].startswith('***'):
        api_keys['clipdrop'] = data['clipdrop']
        os.environ['CLIPDROP_API_KEY'] = data['clipdrop']

    if data.get('replicate') and not data['replicate'].startswith('***'):
        api_keys['replicate'] = data['replicate']
        os.environ['REPLICATE_API_TOKEN'] = data['replicate']

    # クラウドAPI利用可能状況を更新
    CLOUD_APIS_AVAILABLE['clipdrop'] = bool(api_keys['clipdrop'])
    CLOUD_APIS_AVAILABLE['replicate'] = bool(api_keys['replicate'])

    return jsonify({'success': True})


@app.route('/inpaint', methods=['POST'])
def inpaint():
    """Inpainting API"""
    try:
        data = request.json

        # 画像とマスクをBase64からデコード
        image = base64_to_cv2(data['image'])
        mask = base64_to_cv2(data['mask'])

        # マスクをグレースケールに変換
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # マスクを二値化（赤色部分を抽出）
        if len(data.get('mask', '')) > 0:
            mask_color = base64_to_cv2(data['mask'])
            # 赤色成分が強い部分をマスクとして抽出
            red_channel = mask_color[:, :, 2]  # BGR -> R
            blue_channel = mask_color[:, :, 0]
            green_channel = mask_color[:, :, 1]
            # 赤が他より明らかに強い部分
            mask = ((red_channel > 100) &
                    (red_channel > blue_channel + 50) &
                    (red_channel > green_channel + 50)).astype(np.uint8) * 255

        method = data.get('method', 'texture_aware')
        patch_size = int(data.get('patch_size', 9))
        search_area = int(data.get('search_area', 50))

        # Inpainting実行
        if method == 'opencv_ns':
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        elif method == 'opencv_telea':
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        elif method == 'lama':
            # LaMa AI Model
            if not AI_MODELS_AVAILABLE.get('lama'):
                return jsonify({
                    'success': False,
                    'error': 'LaMaがインストールされていません。pip install simple-lama-inpainting'
                }), 400
            from ai_inpaint import LamaInpainter
            inpainter = LamaInpainter()
            result = inpainter.inpaint(image, mask)
        elif method == 'sd':
            # Stable Diffusion
            if not AI_MODELS_AVAILABLE.get('sd'):
                return jsonify({
                    'success': False,
                    'error': 'Stable Diffusionがインストールされていません。pip install diffusers transformers accelerate torch'
                }), 400
            from ai_inpaint import StableDiffusionInpainter
            inpainter = StableDiffusionInpainter()
            prompt = data.get('prompt', 'background pattern, seamless texture')
            result = inpainter.inpaint(image, mask, prompt=prompt)
        elif method == 'clipdrop':
            # ClipDrop API
            if not CLOUD_APIS_AVAILABLE.get('clipdrop'):
                return jsonify({
                    'success': False,
                    'error': 'ClipDrop APIキーが設定されていません。環境変数 CLIPDROP_API_KEY を設定してください。'
                }), 400
            from cloud_inpaint import ClipDropInpainter
            inpainter = ClipDropInpainter()
            result = inpainter.cleanup(image, mask)
        elif method == 'replicate':
            # Replicate API
            if not CLOUD_APIS_AVAILABLE.get('replicate'):
                return jsonify({
                    'success': False,
                    'error': 'Replicate APIトークンが設定されていません。環境変数 REPLICATE_API_TOKEN を設定してください。'
                }), 400
            from cloud_inpaint import ReplicateInpainter
            inpainter = ReplicateInpainter()
            result = inpainter.inpaint_lama(image, mask)
        else:  # texture_aware
            inpainter = TextureAwareInpainter(patch_size, search_area)
            result = inpainter.inpaint(image, mask)

        # 結果をBase64に変換
        result_base64 = cv2_to_base64(result)

        return jsonify({
            'success': True,
            'result': result_base64
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models', methods=['GET'])
def get_models():
    """利用可能なモデル一覧を返す"""
    print(f"[/models] AI_MODELS_AVAILABLE = {AI_MODELS_AVAILABLE}")
    print(f"[/models] id(AI_MODELS_AVAILABLE) = {id(AI_MODELS_AVAILABLE)}")
    return jsonify({
        'available': AI_MODELS_AVAILABLE,
        'cloud_available': CLOUD_APIS_AVAILABLE,
        'all_models': ['opencv_ns', 'opencv_telea', 'texture_aware', 'lama', 'sd', 'clipdrop', 'replicate']
    })


@app.route('/demo', methods=['GET'])
def generate_demo():
    """デモ用のテスト画像とマスクを生成（元の投稿と同じ斜線パターン）"""
    width, height = 400, 300

    # 白背景
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 斜線パターンを描画（元の投稿と同じスタイル）
    line_color = (100, 100, 100)  # グレーの線
    spacing = 15  # 線の間隔

    # 右上がりの斜線
    for i in range(-height, width + height, spacing):
        cv2.line(image, (i, height), (i + height, 0), line_color, 1)

    # 棒人間を描画
    person_color = (0, 0, 0)  # 黒
    cx, cy = width // 2, height // 2

    # 頭
    cv2.circle(image, (cx, cy - 50), 18, person_color, 2)
    # 体
    cv2.line(image, (cx, cy - 32), (cx, cy + 20), person_color, 2)
    # 腕
    cv2.line(image, (cx, cy - 15), (cx - 35, cy + 10), person_color, 2)
    cv2.line(image, (cx, cy - 15), (cx + 35, cy + 10), person_color, 2)
    # 足
    cv2.line(image, (cx, cy + 20), (cx - 25, cy + 65), person_color, 2)
    cv2.line(image, (cx, cy + 20), (cx + 25, cy + 65), person_color, 2)

    # マスクを生成（棒人間の部分）
    mask = np.zeros((height, width), dtype=np.uint8)
    # 頭
    cv2.circle(mask, (cx, cy - 50), 25, 255, -1)
    # 体
    cv2.line(mask, (cx, cy - 32), (cx, cy + 20), 255, 15)
    # 腕
    cv2.line(mask, (cx, cy - 15), (cx - 35, cy + 10), 255, 15)
    cv2.line(mask, (cx, cy - 15), (cx + 35, cy + 10), 255, 15)
    # 足
    cv2.line(mask, (cx, cy + 20), (cx - 25, cy + 65), 255, 15)
    cv2.line(mask, (cx, cy + 20), (cx + 25, cy + 65), 255, 15)

    # Base64に変換
    image_base64 = cv2_to_base64(image)

    # マスクを赤色で可視化
    mask_color = np.zeros((height, width, 3), dtype=np.uint8)
    mask_color[mask > 0] = [0, 0, 255]  # 赤色（BGR）
    mask_base64 = cv2_to_base64(mask_color)

    return jsonify({
        'success': True,
        'image': image_base64,
        'mask': mask_base64,
        'width': width,
        'height': height
    })


@app.route('/compare', methods=['POST'])
def compare():
    """全手法で比較"""
    try:
        data = request.json

        image = base64_to_cv2(data['image'])
        mask_color = base64_to_cv2(data['mask'])

        # 赤色マスク抽出
        red = mask_color[:, :, 2]
        blue = mask_color[:, :, 0]
        green = mask_color[:, :, 1]
        mask = ((red > 100) & (red > blue + 50) & (red > green + 50)).astype(np.uint8) * 255

        results = {}

        # OpenCV NS
        results['opencv_ns'] = cv2_to_base64(
            cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        )

        # OpenCV Telea
        results['opencv_telea'] = cv2_to_base64(
            cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        )

        # テクスチャ対応
        inpainter = TextureAwareInpainter(patch_size=9, search_area=50)
        results['texture_aware'] = cv2_to_base64(
            inpainter.inpaint(image, mask)
        )

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("テクスチャ対応 Inpainting Web App")
    print("=" * 50)
    print("\nブラウザで http://localhost:5001 を開いてください")
    print("終了: Ctrl+C")
    print("=" * 50)
    print(f"AI Models: {AI_MODELS_AVAILABLE}")
    print(f"Cloud APIs: {CLOUD_APIS_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
