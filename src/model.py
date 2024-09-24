# model.py
import pandas as pd
from pycaret.classification import *
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def load_data(train_path, test_path):
    """Verileri yükler.""" 
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def setup_pycaret(data, target):
    """PyCaret oturumunu başlatır.""" 
    return setup(data=data, target=target, session_id=123, fold=5)

def create_and_finalize_model(model_name):
    """Belirtilen modeli PyCaret ile oluşturur ve finalize eder.""" 
    model = create_model(model_name)
    return finalize_model(model)

def make_predictions(model, test_data):
    """Model ile test verisinde tahmin yapar.""" 
    return predict_model(model, data=test_data)

def evaluate_model_with_kfolds(model, X, y, cv=5):
    """K-Folds ile modelin başarı metriklerini döner.""" 
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return f1_scores.max()

def select_best_model_with_kfolds(train_data, target_column):
    """XGBoost ve CatBoost modellerini K-Folds ile değerlendirir ve en iyisini seçer.""" 
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]

    best_model_name = None
    best_f1_score = -float('inf')

    # XGBoost modeli K-Folds ile değerlendir
    xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgboost_f1_score = evaluate_model_with_kfolds(xgboost_model, X, y)

    # CatBoost modeli K-Folds ile değerlendir
    catboost_model = CatBoostClassifier(silent=True)
    catboost_f1_score = evaluate_model_with_kfolds(catboost_model, X, y)

    # Sonuçları kıyasla ve en iyi modeli seç
    if xgboost_f1_score > best_f1_score:
        best_f1_score = xgboost_f1_score
        best_model_name = 'xgboost'

    if catboost_f1_score > best_f1_score:
        best_f1_score = catboost_f1_score
        best_model_name = 'catboost'

    return best_model_name

def save_submission(predictions, test_data, submission_path):
    """Tahminleri ve test verisini kullanarak sonuçları kaydeder.""" 
    submission = pd.DataFrame({
        'id': test_data['id'],
        'value_increased': predictions['prediction_label']  # 'Label' yerine 'prediction_label' kullan
    })
    submission.to_csv(submission_path, index=False)
    print("Sonuçlar kaydedildi.")

def main():
    # Verileri yükle
    train_data, test_data = load_data(r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv', 
                                      r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_test.csv')

    # PyCaret oturumunu başlat
    setup_pycaret(train_data, 'value_increased')

    # PyCaret modellerini oluştur ve finalize et
    models = {
        'catboost': create_and_finalize_model('catboost'),
        'xgboost': create_and_finalize_model('xgboost')
    }

    # K-Folds ile XGBoost ve CatBoost modellerini değerlendir
    best_model_name = select_best_model_with_kfolds(train_data, 'value_increased')

    # En iyi modelin tahminlerini yap ve CSV dosyasına kaydet
    if best_model_name == 'xgboost':
        best_predictions = make_predictions(models['xgboost'], test_data)
    elif best_model_name == 'catboost':
        best_predictions = make_predictions(models['catboost'], test_data)

    # Tahmin sütun adlarını kontrol et ve kaydet
    save_submission(best_predictions, test_data, r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\sample_submission.csv')

if __name__ == "__main__":
    main()
