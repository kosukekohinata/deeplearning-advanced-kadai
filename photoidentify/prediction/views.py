from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import os

# VGG16用の関数をインポート（ここから3行追加）
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np


def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            
            # --- 変更点①：画像の前処理 ---
            # VGG16の入力サイズである224x224に変更
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            # 判定できるように次元を増やす
            img_array = np.expand_dims(img_array, axis=0)
            # VGG16専用の前処理を実行
            processed_img = preprocess_input(img_array)
            
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
            model = load_model(model_path)
            
            # 前処理済みの画像をモデルに渡して予測
            result = model.predict(processed_img)
            
            # --- 変更点②：予測結果の解析 ---
            # VGG16の1000分類の予測結果を、人間が分かりやすいように上位5件に絞って解読
            decoded_result = decode_predictions(result, top=5)[0]
            
            # HTMLで表示しやすいように整形
            predictions = []
            for class_id, name, score in decoded_result:
                # 確率をパーセント表示に変換
                score_percent = f"{score*100:.2f}%"
                predictions.append({'name': name, 'score': score_percent})

            img_data = request.POST.get('img_data')
            
            return render(request, 'home.html', {
                'form': form, 
                'predictions': predictions,  # 辞書型のリストをテンプレートに渡す
                'img_data': img_data
            })
            
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})