# Veri Ön İşleme Raporu

Bu rapor, veri ön işleme sürecinde kullanılan adımları detaylandırmaktadır. Kod, veri yüklemeden eksik değer doldurmaya, kategorik verilerin kodlanmasından aykırı değerlerin temizlenmesine kadar birçok aşamayı kapsamaktadır.

## 1. Veri Yükleme
**Fonksiyon:** `load_data(file_path)`
- **Açıklama:** CSV dosyasını yükler ve bir Pandas DataFrame olarak döner.
- **Kullanım:** Veri seti dosya yolunu belirterek yükleme işlemi yapılır.

## 2. Eksik Değerleri Doldurma ve Veri Temizleme
**Fonksiyon:** `clean_data(df)`
- **Açıklama:** Eksik değerleri doldurur ve veri temizleme adımlarını uygular. 
  - Yükseklik (`Height`) ve ağırlık (`Weight`) verilerindeki birimlerin ('CM' ve 'KG') kaldırılmasını sağlar.
  - Sürekli değişkenlerde (örneğin `Height`, `Weight`) medyan ile, kategorik değişkenlerde ise mod ile eksik değerler doldurulur.
- **Yöntem:** 
  - **Sürekli Değişkenler:** Medyan ile doldurulmuş.
  - **Kategorik Değişkenler:** Mod (en sık görülen değer) ile doldurulmuş.

## 3. Kategorik Verilerin Kodlanması
**Fonksiyon:** `encode_categorical(df)`
- **Açıklama:** Kategorik verileri sayısal değerlere dönüştürür. Örneğin, `Foot` değişkenindeki sağ/sol ayak bilgisi 0 ve 1 olarak kodlanır.
- **Yöntem:** `LabelEncoder` kullanılarak sayısal değerler atanır.

## 4. Ölçeklendirme (Normalization)
**Fonksiyon:** `scale_data(df, columns)`
- **Açıklama:** Sayısal verileri belirli bir aralığa (0-1) ölçeklendirir. Bu, modelleme aşamasında verilerin aynı ölçekte olmasını sağlar.
- **Yöntem:** `MinMaxScaler` kullanılarak veriler 0 ile 1 arasında ölçeklendirilir.

## 5. Aykırı Değerlerin Kontrolü
**Fonksiyon:** `remove_outliers(df, column)`
- **Açıklama:** IQR (Çeyrekler Arası Aralık) yöntemiyle aykırı değerleri belirler ve çıkarır. Özellikle aşırı büyük veya küçük değerlerin etkisini azaltmak için kullanılır.
- **Yöntem:** Aykırı değerler belirlenir ve çıkarılır.

## 6. Yeni Özellikler Üretme
**Fonksiyon:** `feature_engineering(df)`
- **Açıklama:** Yeni özellikler üretir. Örneğin, `BMI` (vücut kitle indeksi) gibi yeni değişkenler eklenir.
- **Yöntem:** `BMI`, ağırlık ve yükseklik kullanılarak hesaplanır.

## 7. Dengesiz Verileri Dengeleme (Opsiyonel)
**Fonksiyon:** `balance_data(df, target_column)`
- **Açıklama:** SMOTE kullanarak dengesiz sınıfları dengeler. Sınıflar arasında ciddi bir dengesizlik varsa, modelin daha iyi öğrenebilmesi için kullanılır.
- **Yöntem:** SMOTE ile yeniden örnekleme yapılır ve yeniden örneklenmiş veri oluşturulur.

## 8. Ana Veri Ön İşleme Fonksiyonu
**Fonksiyon:** `preprocess_data(df)`
- **Açıklama:** Tüm veri ön işleme adımlarını çalıştırır:
  - Eksik değerlerin doldurulması ve birim dönüşümü
  - Kategorik verilerin kodlanması
  - Aykırı değerlerin çıkarılması
  - Özellik mühendisliği (BMI ekleme)
  - Ölçeklendirme (Height, Weight ve BMI sütunları)
- **Yöntem:** Yukarıda belirtilen fonksiyonlar sırayla uygulanır.
