import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

# Log ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path):
    """Verileri yükler."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Eğitim ve test verilerinden id sütununu çıkarıyoruz
    X_train = train_data.drop(columns=['value_increased', 'id'])
    y_train = train_data['value_increased']
    X_test = test_data.drop(columns=['id'])
    
    return X_train, y_train, X_test, test_data

def smote_oversample(X, y):
    """SMOTE ile sınıf dengesizliği giderilir."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def evaluate_model_with_kfolds(model, X, y, cv=5):
    """K-Folds ile modelin başarı metriklerini döner."""
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    logging.info(f"{model.__class__.__name__} - F1 Skorları: {f1_scores}")
    return f1_scores.mean()

def make_predictions(model, test_data):
    """Model ile test verisinde tahmin yapar."""
    predictions = model.predict(test_data)
    return predictions

def save_submission(predictions, test_data, submission_path):
    """Tahminleri ve test verisini kullanarak sonuçları kaydeder."""
    submission = pd.DataFrame({
        'id': test_data['id'],  # id'yi koruyoruz çünkü gönderim dosyasında olmalı
        'value_increased': predictions
    })
    submission.to_csv(submission_path, index=False)
    logging.info(f"Sonuçlar {submission_path} dosyasına kaydedildi.")

def hyperparameter_optimization(X, y):
    """RandomizedSearchCV ile hyperparametre optimizasyonu."""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    
    logging.info(f"En iyi parametreler: {random_search.best_params_}")
    return random_search.best_estimator_

def train_and_evaluate_models(X, y):
    """Birden fazla modeli eğit ve değerlendir."""
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }
    
    model_scores = {}
    
    for name, model in models.items():
        logging.info(f"{name} modeli eğitiliyor...")
        score = evaluate_model_with_kfolds(model, X, y)
        model_scores[name] = score
        logging.info(f"{name} - Ortalama F1 Skoru: {score}")
    
    return models, model_scores

def main():
    # Verileri yükle
    X_train, y_train, X_test, test_data = load_data(
        r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv',
        r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_test.csv'
    )

    # SMOTE ile sınıf dengesizliği giderme
    X_train_res, y_train_res = smote_oversample(X_train, y_train)

    # Hyperparametre optimizasyonu
    best_rf_model = hyperparameter_optimization(X_train_res, y_train_res)

    # Tahmin yap
    best_rf_model.fit(X_train_res, y_train_res)
    predictions = make_predictions(best_rf_model, X_test)

    # Sonuçları kaydet
    save_submission(predictions, test_data, r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\best_model_submission.csv')

if __name__ == "__main__":
    main()
