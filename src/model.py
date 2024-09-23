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

def print_metrics(model_name, metrics):
    """Başarı metriklerini ekrana basar.""" 
    print(f"\n{model_name} Başarı Metrikleri:")
    if 'Accuracy' in metrics.index:
        print(f"Accuracy: {metrics.loc['Accuracy', 'Mean']:.4f}")
    if 'F1' in metrics.index:
        print(f"F1 Score: {metrics.loc['F1', 'Mean']:.4f}")
    if 'Prec.' in metrics.index:
        print(f"Precision: {metrics.loc['Prec.', 'Mean']:.4f}")
    if 'Recall' in metrics.index:
        print(f"Recall: {metrics.loc['Recall', 'Mean']:.4f}")

def evaluate_and_print_metrics(models, test_data):
    """PyCaret modellerini değerlendirir ve başarı metriklerini ekrana basar.""" 
    best_model_name = None
    best_f1_score = -float('inf')

    for model_name, model in models.items():
        print(f"\n{model_name} modeli değerlendiriliyor...")
        predictions = make_predictions(model, test_data)
        metrics = pull()  # PyCaret'ten model metriklerini al

        print_metrics(model_name, metrics)

        # En yüksek F1 skorunu takip et
        if metrics.loc['F1'].max() > best_f1_score:
            best_f1_score = metrics.loc['F1'].max()
            best_model_name = model_name

    return best_model_name, best_f1_score

def evaluate_model_with_kfolds(model, X, y, cv=5):
    """K-Folds ile modelin başarı metriklerini döner.""" 
    print(f"\n{model.__class__.__name__} modeli K-Folds ile değerlendiriliyor...")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    max_f1 = f1_scores.max()
    print(f"{model.__class__.__name__} F1 Score (K-Folds): {max_f1:.4f}")
    return max_f1

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

    return best_model_name, best_f1_score

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

    # PyCaret modellerini oluştur ve finalize et
    models = {
        'catboost': create_and_finalize_model('catboost'),
        'xgboost': create_and_finalize_model('xgboost')
    }

    # PyCaret modellerini değerlendir ve en iyi modeli seç
    best_model_name_pycaret, best_f1_pycaret = evaluate_and_print_metrics(models, test_data)

    # K-Folds ile XGBoost ve CatBoost modellerini değerlendir
    best_model_name_kfolds, best_f1_kfolds = select_best_model_with_kfolds(train_data, 'value_increased')

    # En iyi modeli güncelle
    best_model_name = best_model_name_pycaret if best_model_name_pycaret else best_model_name_kfolds
    best_f1_score = best_f1_pycaret if best_model_name_pycaret else best_f1_kfolds

    print(f"\nEn iyi model (PyCaret): {best_model_name_pycaret} (F1 Score: {best_f1_pycaret:.4f})")
    print(f"En iyi model (K-Folds): {best_model_name_kfolds} (F1 Score: {best_f1_kfolds:.4f})")
    print(f"Güncellenmiş En iyi model: {best_model_name} (F1 Score: {best_f1_score:.4f})")

    # En iyi modelin tahminlerini yap ve CSV dosyasına kaydet
    if best_model_name == 'xgboost':
        best_predictions = make_predictions(models['xgboost'], test_data)
    elif best_model_name == 'catboost':
        best_predictions = make_predictions(models['catboost'], test_data)

    save_submission(best_predictions, test_data, r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\sample_submission.csv')

if __name__ == "__main__":
    main()
