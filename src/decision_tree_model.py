import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging
import warnings

# Sadece hata mesajlarını göster
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Uyarıları tamamen bastır
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train = train_data.drop(columns=['value_increased', 'id'])
    y_train = train_data['value_increased']
    X_test = test_data.drop(columns=['id'])
    
    return X_train, y_train, X_test, test_data

def create_balanced_model(model, sampler=SMOTE(random_state=42)):
    return Pipeline([
        ('sampler', sampler),
        ('classifier', model)
    ])

def evaluate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1_scores.append(f1_score(y_val, y_pred))
    
    return np.mean(f1_scores)

def create_ensemble_model():
    models = [
        ('rf', RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(scale_pos_weight=10, random_state=42)),
        ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1))  # verbose=-1 eklendi
    ]
    return VotingClassifier(estimators=models, voting='soft')

def adjust_threshold(y_proba, threshold=0.2):
    return (y_proba >= threshold).astype(int)

def main():
    train_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv'
    test_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_test.csv'
    X_train, y_train, X_test, test_data = load_and_preprocess_data(train_path, test_path)
    
    logging.info("Data loaded and preprocessed.")
    
    # Create and evaluate balanced models
    models = [
        create_balanced_model(RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)),
        create_balanced_model(GradientBoostingClassifier(random_state=42)),
        create_balanced_model(XGBClassifier(scale_pos_weight=10, random_state=42)),
        create_balanced_model(LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1))  # verbose=-1 eklendi
    ]
    
    for model in models:
        f1 = evaluate_model(model, X_train, y_train)
        logging.info(f"{model['classifier'].__class__.__name__} F1 Score: {f1}")
    
    # Create and train ensemble model
    ensemble = create_ensemble_model()
    ensemble.fit(X_train, y_train)
    logging.info("Ensemble model trained.")
    
    # Make predictions with lower threshold
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = adjust_threshold(y_proba, threshold=0.2)
    
    # Save results
    submission = pd.DataFrame({'id': test_data['id'], 'value_increased': y_pred})
    submission_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\ensemble_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    # Log completion message
    logging.info(f"Results saved to {submission_path}")
    
    # Log class distribution in predictions
    pred_distribution = pd.Series(y_pred).value_counts()
    logging.info(f"Prediction distribution:\n{pred_distribution}")

if __name__ == "__main__":
    main()
