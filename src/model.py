# model.py
import pandas as pd
from pycaret.classification import *

def load_data(train_path, test_path):
    """Verileri yükler."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def setup_pycaret(data, target):
    """PyCaret oturumunu başlatır."""
    return setup(data=data, target=target, session_id=123)

def compare_models():
    """Modelleri karşılaştırır ve en iyisini döner."""
    return compare_models()

def create_and_finalize_model(model_name):
    """Belirtilen modeli oluşturur ve finalize eder."""
    model = create_model(model_name)
    return finalize_model(model)

def make_predictions(model, test_data):
    """Model ile test verisinde tahmin yapar."""
    return predict_model(model, data=test_data)

def evaluate_models(models, test_data):
    """Modelleri değerlendirir ve performanslarını döner."""
    results = {}
    for model_name, model in models.items():
        print(f"Model: {model_name} değerlendirme yapılıyor...")
        predictions = make_predictions(model, test_data)
        results[model_name] = predictions
    return results

def select_best_model(results):
    """En iyi modeli seçmek için tahmin sonuçlarını değerlendirir."""
    # Burada hangi metriklere bakarak karar vereceğimize karar verebiliriz (accuracy, f1, vs.)
    # Örnek olarak basitçe ilk modeli seçiyoruz. Geliştirebiliriz.
    best_model_name = list(results.keys())[0]
    return best_model_name, results[best_model_name]

def save_submission(predictions, test_data, submission_path):
    """Tahminleri ve test verisini kullanarak sonuçları kaydeder."""
    submission = pd.DataFrame({
        'id': test_data['id'],
        'value_increased': predictions['Label']
    })
    submission.to_csv(submission_path, index=False)
    print("Sonuçlar kaydedildi.")

def main():
    # Verileri yükle
    train_data, test_data = load_data(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv', r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_test.csv')

    # PyCaret oturumunu başlat
    setup_pycaret(train_data, 'value_increased')

    # CatBoost ve XGBoost ile model oluştur ve finalize et
    models = {
        'catboost': create_and_finalize_model('catboost'),
        'xgboost': create_and_finalize_model('xgboost')
    }

    # Modelleri değerlendir
    results = evaluate_models(models, test_data)

    # En iyi modeli seç
    best_model_name, best_predictions = select_best_model(results)

    print(f"En iyi model: {best_model_name}")

    # Sonuçları kaydetmeden önce en uygun model olup olmadığını onayla
    confirm = input("Sonuçları kaydetmek istiyor musunuz? (evet/hayır): ").lower()
    if confirm == 'evet':
        save_submission(best_predictions, test_data, 'data/sample_submission.csv')

if __name__ == "__main__":
    main()
