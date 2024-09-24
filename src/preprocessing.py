import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE
import logging
import warnings

# Uyarı mesajlarını bastır
warnings.filterwarnings('ignore', category=Warning)

# Bilgi mesajlarını yapılandır
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Veri Yükleme Fonksiyonu
def load_data(file_path):
    """ CSV dosyasını yükler ve bir DataFrame olarak döner."""
    logging.info("Veri yükleniyor...")
    return pd.read_csv(file_path)

# 2. Eksik Değerleri KNN ile Doldurma ve Veri Temizleme
def clean_data_knn(df):
    logging.info("Veri KNN ile temizleniyor...")

    # Yükseklik ve ağırlık birimlerini temizleme ve sayısal hale dönüştürme
    df['Height'] = df['Height'].apply(lambda x: str(x).replace(' CM', '') if isinstance(x, str) else x)
    df['Weight'] = df['Weight'].apply(lambda x: str(x).replace(' KG', '') if isinstance(x, str) else x)

    # 'Height' ve 'Weight' sütunlarını sayısal veri tipine dönüştürme
    df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

    # Kategorik değişkenleri kodlama öncesi eksik değerleri doldurmak için
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

    # Kategorik değişkenleri sayısal hale getirme (Label Encoding)
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # KNN Imputer için sayısal sütunları seçme
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # KNN Imputer uygulama
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

# 4. Ölçeklendirme (Normalization)
def scale_data(df, columns):
    logging.info("Veriler ölçeklendiriliyor...")
    scaler = MinMaxScaler()
    df.loc[:, columns] = scaler.fit_transform(df[columns])
    return df

# 6. Yeni Özellikler Üretme
def feature_engineering(df):
    logging.info("Yeni özellikler üretiliyor...")
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    return df

# 7. Dengesiz Verileri Dengeleme
def balance_data(df, target_column):
    logging.info("Veriler SMOTE ile dengeleniyor...")
    smote = SMOTE()
    X = df.drop(target_column, axis=1)  # Hedef sütunu hariç tüm sütunlar
    y = df[target_column]  # Hedef sütun (bağımlı değişken)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # SMOTE ile yeniden örnekleme
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)  # Yeniden örneklenmiş veriyi birleştir
    return df_resampled

# 8. Ana Veri Ön İşleme Fonksiyonu
def preprocess_data(df, balance=False, target_column=None):
    df = clean_data_knn(df)
    
    # Özellik mühendisliği (BMI ekleme)
    df = feature_engineering(df)

    # Ölçeklendirme (Height, Weight ve BMI sütunlarını ölçeklendiriyoruz)
    df = scale_data(df, ['Height', 'Weight', 'BMI'])

    # Opsiyonel: Veri dengesizse SMOTE ile dengeleme
    if balance and target_column:
        df = balance_data(df, target_column)

    return df

# 9. İşlenmiş Verilerin Kaydedilmesi
if __name__ == "__main__":
    # Dosya yolları
    train_data_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\train.csv'
    train_save_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv'
    test_data_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\test.csv'
    test_save_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_test.csv'

    # Eğitim verisini yükleme ve işleme
    train_df = load_data(train_data_path)
    train_df_processed = preprocess_data(train_df, balance=True, target_column='value_increased')
    train_df_processed.to_csv(train_save_path, index=False)
    logging.info("Eğitim verisi başarıyla işlendi ve kaydedildi.")

    # Test verisini yükleme ve işleme
    test_df = load_data(test_data_path)
    test_df_processed = preprocess_data(test_df)
    test_df_processed.to_csv(test_save_path, index=False)
    logging.info("Test verisi başarıyla işlendi ve kaydedildi.")
