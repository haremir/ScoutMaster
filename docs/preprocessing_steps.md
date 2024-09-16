# Veri Ön İşleme Adımları

## 1. Eksik Veri Doldurma
Veri setindeki eksik değerler `0` ile dolduruldu.

## 2. Yükseklik ve Ağırlık Birimlerinin Düzeltilmesi
- `Height`: Veride 'CM' birimi kaldırılarak tam sayı değerine dönüştürüldü.
- `Weight`: Veride 'KG' birimi kaldırılarak tam sayı değerine dönüştürüldü.

## 3. Kategorik Değerlerin Sayısallaştırılması
- `Foot`: Sağ/Sol ayak kullanımı `LabelEncoder` ile sayısal değerlere dönüştürüldü.
