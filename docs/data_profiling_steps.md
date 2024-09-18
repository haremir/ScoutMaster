# Veri Profil Çıkarımı ve Analizi

Bu bölümde, veri setlerimizin genel özetini, değişken analizlerini, korelasyonlarını ve eksik değer analizlerini **ydata-profiling** kullanarak çıkardık. Analizler, hem ham hem de işlenmiş veriler üzerinde yapılmıştır ve her bir veri seti için ayrı ayrı profil raporları oluşturulup HTML dosyaları olarak kaydedilmiştir.

## 1. Kullanılan Yöntem ve Araçlar
- **Pandas Profiling (ydata-profiling)**: Veri setlerinin hızlı ve kapsamlı özet raporlarını çıkaran Python kütüphanesi.
- **Profil Raporu Çıktıları**: Raporlar, veri setlerinin genel istatistikleri, eksik değer analizi, dağılım grafikleri, korelasyonlar ve daha fazlasını içerir.

## 2. Profil Çıkarılan Veri Setleri
Profil raporları aşağıdaki veri setleri için oluşturulmuştur:
1. **Ham Eğitim Verisi (train.csv)**
2. **Ham Test Verisi (test.csv)**
3. **İşlenmiş Eğitim Verisi (processed_train.csv)**
4. **İşlenmiş Test Verisi (processed_test.csv)**

## 3. Profil Oluşturma Adımları
Her veri seti için aşağıdaki adımlar izlenmiştir:

### Veri Setinin Yüklenmesi
Veri setleri, Pandas kütüphanesi kullanılarak yüklenmiştir.

### Profil Raporlarının Oluşturulması
Her veri seti için profil raporları ydata-profiling aracılığıyla oluşturulmuş ve HTML formatında kaydedilmiştir.

**Çıktılar:**
Aşağıdaki HTML dosyaları oluşturulmuş ve `reports/profiling/` dizinine kaydedilmiştir:
- `train_raw_profil_raporu.html`
- `test_raw_profil_raporu.html`
- `train_processed_profil_raporu.html`
- `test_processed_profil_raporu.html`

## Analiz Özeti
Oluşturulan raporlar, aşağıdaki başlıklar altında veri setlerinin derinlemesine analizini sunmaktadır:
- **Genel Özet**: Veri setindeki toplam satır sayısı, eksik değerlerin oranı, değişken türleri ve temel istatistikler.
- **Değişken Analizi**: Her bir değişkenin dağılımı, olası aykırı değerler, veri tipi gibi detaylı analizler.
- **Korelasyon Analizi**: Sayısal değişkenler arasındaki korelasyonlar ve görselleştirilmiş korelasyon matrisi.
- **Eksik Değer Analizi**: Eksik veri bulunan kolonlar ve bu verilerin genel dağılımı.

## Sonuç
Veri setlerinin profil raporları, veri temizleme ve modelleme sürecinin ilerleyen aşamalarında önemli kararlar almak için kullanılmıştır. Bu raporlar, eksik değerlerin dağılımı, aykırı değerlerin tespiti ve korelasyonların gözlenmesini kolaylaştırmıştır.
