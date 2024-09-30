# Ensemble Model ile Tahmin

Bu rapor, projemin son aşamalarını ve uyguladığım yöntemleri özetlemektedir. Projem, veri ön işleme, model geliştirme ve sonuçların değerlendirilmesi adımlarını içermektedir. Kullanılan algoritmalar arasında Karar Ağaçları, XGBoost, LightGBM ve ensemble modelleme yer almaktadır.

## 1. Veri Yükleme ve Ön İşleme

Veri setleri, `processed_train.csv` ve `processed_test.csv` dosyalarından yüklendi. Aşağıdaki adımlar gerçekleştirildi:

- **Veri Yükleme**: Pandas kullanılarak veriler yüklendi.
- **Hedef Değişkenin Ayrılması**: `value_increased` hedef değişkeni, özelliklerden ayrıldı.
- **Gereksiz Sütunların Kaldırılması**: `id` sütunu çıkarıldı.

## 2. Model Geliştirme

### 2.1. Dengeleme ve Modelleme

Veri dengesizliği nedeniyle, SMOTE (Synthetic Minority Over-sampling Technique) kullanarak model oluşturma sürecinde dengenin sağlanması hedeflendi. Aşağıdaki modeller oluşturuldu ve değerlendirildi:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**
- **LightGBM Classifier**

### 2.2. Değerlendirme

Model performansı, F1 skoru ile değerlendirildi. `StratifiedKFold` yöntemi ile 5 katlı çapraz doğrulama uygulandı.

## 3. Ensemble Model

Çeşitli modellerin bir araya getirildiği bir ensemble model oluşturuldu. Bu modelin yapısı:

- **RandomForestClassifier**
- **GradientBoostingClassifier**
- **XGBClassifier**
- **LGBMClassifier**

Bu modellerin çıktılarını birleştirerek daha güvenilir tahminler elde edilmesi amaçlandı.

## 4. Tahmin Yapma

Ensemble model ile test veri setine tahmin yapıldı. Tahmin sonuçlarının sınıf dağılımı izlendi ve sonuçlar, belirli bir eşik değerine göre ayarlandı.

## 5. Sonuçların Kaydedilmesi

Tahmin sonuçları `best_submission.csv` dosyasına kaydedildi. Bu dosyada, her bir test örneğine karşılık gelen tahmin değerleri ve id’leri bulunmaktadır.

## 6. Sonuçlar ve Gelecek Aşamalar

Proje tamamlandıktan sonra elde edilen sonuçlar değerlendirilecek ve ileride yapılabilecek geliştirmeler belirlenecektir. Bu aşamada projenin çıktıları, Kaggle yarışmasında elde edilen başarı ile de ilgili olacaktır.

