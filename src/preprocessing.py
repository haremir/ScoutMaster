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
    """
    Eksik değerleri KNN algoritması ile doldurur ve gerekli veri temizleme adımlarını uygular.
    """
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

# 3. Kategorik Verilerin Kodlanması
def encode_categorical(df):
    """
    Kategorik değişkenleri sayısal hale getirir.
    Örneğin, 'Foot' değişkeni için sağ/sol ayak bilgisi 0 ve 1 olarak kodlanır.
    """
    # Bu adımı clean_data_knn içinde yaptığımız için tekrar yapmamıza gerek yok
    return df

# 4. Ölçeklendirme (Normalization)
def scale_data(df, columns):
    """
    Sayısal verileri belirli bir aralığa (0-1) ölçeklendirir.
    Özellikle modelleme aşamasında verilerin aynı ölçekte olmasını sağlamak için gereklidir.
    """
    logging.info("Veriler ölçeklendiriliyor...")
    scaler = MinMaxScaler()
    df.loc[:, columns] = scaler.fit_transform(df[columns])
    return df

# 5. Aykırı Değerlerin Kontrolü
def remove_outliers(df, column):
    """
    IQR yöntemiyle aykırı değerleri belirler ve çıkarır.
    Özellikle aşırı büyük veya küçük değerlerin modeli olumsuz etkilemesini önlemek için kullanılır.
    """
    logging.info(f"Aykırı değerler çıkarılıyor: {column}...")
    Q1 = df[column].quantile(0.25)  # 1. Çeyrek
    Q3 = df[column].quantile(0.75)  # 3. Çeyrek
    IQR = Q3 - Q1  # Çeyrekler arası fark
    lower_bound = Q1 - 1.5 * IQR  # Alt sınır
    upper_bound = Q3 + 1.5 * IQR  # Üst sınır
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]  # Aykırı değerleri çıkar
    return df

# 6. Yeni Özellikler Üretme
def feature_engineering(df):
    """
    Yeni özellikler üretir. Örneğin, 'BMI' (vücut kitle indeksi) gibi yeni değişkenler eklenir.
    Yeni özellikler, modelin daha iyi öğrenmesini sağlayabilir.
    """
    logging.info("Yeni özellikler üretiliyor...")
    # BMI (Vücut Kitle İndeksi) hesaplama
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    return df

# 7. Dengesiz Verileri Dengeleme
def balance_data(df, target_column):
    """
    SMOTE kullanarak dengesiz sınıfları dengeler.
    Özellikle sınıflar arasında ciddi bir dengesizlik varsa, modelin daha iyi öğrenebilmesi için kullanılır.
    """
    logging.info("Veriler SMOTE ile dengeleniyor...")
    smote = SMOTE()
    X = df.drop(target_column, axis=1)  # Hedef sütunu hariç tüm sütunlar
    y = df[target_column]  # Hedef sütun (bağımlı değişken)
    X_resampled, y_resampled = smote.fit_resample(X, y)  # SMOTE ile yeniden örnekleme
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)  # Yeniden örneklenmiş veriyi birleştir
    return df_resampled

# 8. Ana Veri Ön İşleme Fonksiyonu
def preprocess_data(df, balance=False, target_column=None):
    """
    Tüm veri ön işleme adımlarını çalıştırır.
    Bu adımlar: eksik değer doldurma, kategorik verilerin kodlanması, aykırı değerlerin çıkarılması,
    ölçeklendirme ve yeni özellikler üretmeyi içerir.
    """
    # 1. Temizlik adımları (eksik değer doldurma ve birim dönüşümü)
    df = clean_data_knn(df)

    # 2. Kategorik verileri kodlama (clean_data_knn içinde yapıldı)

    # 3. Aykırı değerlerin çıkarılması (örnek: Height)
    df = remove_outliers(df, 'Height')

    # 4. Özellik mühendisliği (BMI ekleme)
    df = feature_engineering(df)

    # 5. Ölçeklendirme (Height, Weight ve BMI sütunlarını ölçeklendiriyoruz)
    df = scale_data(df, ['Height', 'Weight', 'BMI'])

    # 6. Opsiyonel: Veri dengesizse SMOTE ile dengeleme
    if balance and target_column:
        df = balance_data(df, target_column)

    return df

# 9. İşlenmiş Verilerin Kaydedilmesi
if __name__ == "__main__":
    # Dosya yolları
    data_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\raw_data\train.csv'
    save_path = r'C:\Users\emirh\Desktop\DOSYALAR\veri_bilimi\scoutmaster\data\processed_data\processed_train.csv'

    # Veriyi yükleme
    df = load_data(data_path)

    # Veri ön işleme (SMOTE opsiyonel olarak etkinleştirilebilir)
    df_processed = preprocess_data(df, balance=True, target_column='value_increased')

    # İşlenmiş veriyi kaydetme
    df_processed.to_csv(save_path, index=False)
    logging.info("Veri başarıyla işlendi ve kaydedildi.")
