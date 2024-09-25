# Veri Ön İşleme Raporu

Bu rapor, veri ön işleme sürecinde kullanılan adımları detaylandırmaktadır. Kod, veri yüklemeden eksik değer doldurmaya, kategorik verilerin kodlanmasından aykırı değerlerin temizlenmesine kadar birçok aşamayı kapsamaktadır.

## 1. Veri Yükleme
**Fonksiyon:** `load_data(file_path)`  
**Açıklama:** CSV dosyasını yükler ve bir Pandas DataFrame olarak döner.  
**Kullanım:** Veri seti dosya yolunu belirterek yükleme işlemi yapılır.

## 2. Eksik Değerleri KNN ile Doldurma ve Veri Temizleme
**Fonksiyon:** `clean_data_knn(df)`  
**Açıklama:** Eksik değerleri KNN algoritması ile doldurur ve veri temizleme adımlarını uygular. 
  - Yükseklik (`Height`) ve ağırlık (`Weight`) verilerindeki birimlerin ('CM' ve 'KG') kaldırılmasını sağlar.
  - Eksik değerler, KNN Imputer ile doldurulur.
**Yöntem:** 
  - **Sürekli Değişkenler:** KNN Imputer ile doldurulmuş.
  - **Kategorik Değişkenler:** En sık görülen değer (mod) ile doldurulmuş.

## 3. Kategorik Verilerin Kodlanması
**Fonksiyon:** `clean_data_knn(df)` (Kodlama işlemi bu fonksiyonun içerisinde yapılır)  
**Açıklama:** Kategorik verileri sayısal değerlere dönüştürür. Örneğin, `Foot` değişkenindeki sağ/sol ayak bilgisi 0 ve 1 olarak kodlanır.  
**Yöntem:** `LabelEncoder` kullanılarak sayısal değerler atanır.

## 4. Ölçeklendirme (Normalization)
**Fonksiyon:** `scale_data(df, columns)`  
**Açıklama:** Sayısal verileri belirli bir aralığa (0-1) ölçeklendirir. Bu, modelleme aşamasında verilerin aynı ölçekte olmasını sağlar.  
**Yöntem:** `MinMaxScaler` kullanılarak veriler 0 ile 1 arasında ölçeklendirilir.

## 5. Aykırı Değerlerin Kontrolü (İsteğe Bağlı)
**Fonksiyon:** Bu adım şu anki sürümde mevcut değil.  
**Açıklama:** Aykırı değerler belirlenip çıkarılmamıştır. Gerekli görüldüğünde IQR (Çeyrekler Arası Aralık) yöntemi kullanılabilir.  
**Yöntem:** İlerleyen sürümlerde eklenmesi planlanıyor.

## 6. Yeni Özellikler Üretme
**Fonksiyon:** `feature_engineering(df)`  
**Açıklama:** Yeni özellikler üretir. Örneğin, `BMI` (vücut kitle indeksi) gibi yeni değişkenler eklenir.  
**Yöntem:** `BMI`, ağırlık ve yükseklik kullanılarak hesaplanır.

## 7. Dengesiz Verileri Dengeleme
**Fonksiyon:** `balance_data(df, target_column)`  
**Açıklama:** SMOTE kullanarak dengesiz sınıfları dengeler. Sınıflar arasında ciddi bir dengesizlik varsa, modelin daha iyi öğrenebilmesi için kullanılır.  
**Yöntem:** SMOTE ile yeniden örnekleme yapılır ve yeniden örneklenmiş veri oluşturulur.

## 8. Ana Veri Ön İşleme Fonksiyonu
**Fonksiyon:** `preprocess_data(df, balance=False, target_column=None)`  
**Açıklama:** Tüm veri ön işleme adımlarını çalıştırır:
  - Eksik değerlerin doldurulması ve birim dönüşümü
  - Kategorik verilerin kodlanması
  - Özellik mühendisliği (BMI ekleme)
  - Ölçeklendirme (Height, Weight ve BMI sütunları)
  - Opsiyonel: SMOTE ile dengeleme
**Yöntem:** Yukarıda belirtilen fonksiyonlar sırayla uygulanır.
