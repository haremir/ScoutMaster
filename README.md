# Futbolcu Değer Tahmin Projesi

## Proje Özeti
Bu proje, futbolcu verilerini analiz ederek, bir futbolcunun bir sonraki sezonda değerinin artıp artmayacağını tahmin etmeyi amaçlamaktadır. Proje, veri ön işleme, özellik mühendisliği, model eğitimi ve değerlendirmesi süreçlerini içermektedir.

## İçindekiler
- [Futbolcu Değer Tahmin Projesi](#futbolcu-değer-tahmin-projesi)
  - [Proje Özeti](#proje-özeti)
  - [İçindekiler](#i̇çindekiler)
  - [Sürümler](#sürümler)
  - [Veri Setleri](#veri-setleri)
  - [Proje Adımları](#proje-adımları)
  - [Kurulum ve Kullanım](#kurulum-ve-kullanım)
  - [Yazarlar](#yazarlar)

## Sürümler
- **V1.0.0**: İlk sürüm; temel model eğitimi ve tahmin süreçlerini içerir.
- **V2.0.0**: Gelecek güncellemelerde daha fazla model eklenmesi ve notebook üzerinden çalışma planlanmaktadır.

## Veri Setleri
Projede kullanılan veri setleri şunlardır:

1. **train.csv**: Futbolcu verileri (eğitim seti)
   - `id`: Futbolcu kimliği
   - `value_increased`: Değer artışı (True/False)
   - `Ability`: Futbolcu yeteneği
   - `Potential`: Potansiyel
   - `Positions`: Oynadığı pozisyonlar
   - `Caps`: Maç sayısı
   - `Goals`: Gol sayısı
   - `Foot`: Hangi ayakla oynadığı
   - `Height`: Boy
   - `Weight`: Ağırlık
   - ... (Diğer özellikler)

## Proje Adımları

1. **Veri Ön İşleme:**
   - Verilerin birleştirilmesi ve temizlenmesi
   - Eksik verilerin işlenmesi

2. **Özellik Mühendisliği:**
   - Özelliklerin oluşturulması ve seçilmesi
   - Verilerin normalize edilmesi

3. **Model Eğitimi:**
   - Seçilen makine öğrenimi algoritmaları kullanılarak model eğitimi
   - Hiperparametre optimizasyonu

4. **Model Değerlendirmesi:**
   - Performans değerlendirmesi

5. **Tahmin:**
   - Test verileri üzerinde tahminler yapılması
   - Tahmin sonuçlarının kaydedilmesi

## Kurulum ve Kullanım

1. **Gereksinimler:**
   - Python 3.x
   - Pandas
   - NumPy
   - Scikit-learn
   - PyCaret
   - Catboost
   - Xgboost

2. **Kurulum:**
   ```bash
   pip install pandas numpy scikit-learn

3. **Projeyi Çalıştırma:**
    - Repo'yu klonlayın:
    git clone https://github.com/username/scoutmaster

    - Veri işleme ve model eğitimi için gerekli scriptsleri çalıştırın:

    ```bash
    python scripts/preprocessing.py
    python scripts/model.py
    ```
## Yazarlar

- Harun Emirhan - [LinkedIn Profilim](https://www.linkedin.com/in/harun-emirhan-bostanci-24144726b/)
