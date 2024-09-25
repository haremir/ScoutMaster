# Makine Öğrenmesi Model Raporu

Bu rapor, makine öğrenmesi modelinin geliştirilmesi sırasında kullanılan adımları ve fonksiyonları detaylandırmaktadır. Model, veri ön işleme, model eğitimi, model değerlendirme ve hiperparametre optimizasyonu gibi aşamaları kapsamaktadır.

## 1. Veri Yükleme ve Bölme
**Fonksiyon:** `load_and_split_data(file_path)`  
**Açıklama:** Veri setini yükler, özellikler (X) ve hedef değişkeni (y) ayırır. Veriyi eğitim ve test olarak böler.  
**Kullanım:** Veri seti dosya yolunu belirterek, eğitim ve test seti oluşturulur.  
**Yöntem:** `train_test_split` kullanılarak verinin %80'i eğitim, %20'si test seti olarak ayrılmıştır.

## 2. Model Kurulumu
**Fonksiyon:** `build_model()`  
**Açıklama:** Logistic Regression modelini oluşturur.  
**Yöntem:** `LogisticRegression` sınıfı kullanılarak model oluşturulmuştur. Standart hiperparametreler ile başlatılmıştır.

## 3. Model Eğitimi
**Fonksiyon:** `train_model(model, X_train, y_train)`  
**Açıklama:** Eğitim verisi kullanılarak modeli eğitir.  
**Kullanım:** Eğitim verisi ve hedef değişkeni modele verilir. Model, bu verilere göre öğrenme işlemini gerçekleştirir.

## 4. Model Tahmini
**Fonksiyon:** `predict_model(model, X_test)`  
**Açıklama:** Eğitim sonrası model, test verisi üzerinde tahminler yapar.  
**Yöntem:** `model.predict(X_test)` fonksiyonu ile test seti üzerindeki tahminler elde edilir.

## 5. Model Değerlendirmesi
**Fonksiyon:** `evaluate_model(model, X_test, y_test)`  
**Açıklama:** Modelin başarımını değerlendirmek için doğruluk (`accuracy`) ve sınıflandırma raporu (`classification report`) hesaplanır.  
**Yöntem:** `accuracy_score` ve `classification_report` kullanılarak modelin doğruluk oranı ve sınıf bazında performansı değerlendirilmiştir.

## 6. Hiperparametre Optimizasyonu (Grid Search)
**Fonksiyon:** `optimize_hyperparameters(model, X_train, y_train)`  
**Açıklama:** Hiperparametreleri en uygun değerleri bulmak için GridSearchCV uygulanır.  
**Yöntem:** `GridSearchCV` kullanılarak farklı hiperparametre kombinasyonları denenmiş ve en iyi model seçilmiştir.

## 7. Sonuçlar
**Fonksiyon:** `final_model_evaluation(model, X_test, y_test)`  
**Açıklama:** Eğitim ve optimizasyon sonrası nihai modelin performansı değerlendirilir. En iyi hiperparametreler ile elde edilen doğruluk oranı ve diğer metrikler raporlanır.  
**Yöntem:** Test seti üzerindeki nihai sonuçlar ve metrikler analiz edilir.

## 8. Ana Model Fonksiyonu
**Fonksiyon:** `train_and_evaluate_model(file_path)`  
**Açıklama:** Tüm model geliştirme adımlarını çalıştırır:
  - Verinin yüklenmesi ve bölünmesi
  - Modelin oluşturulması
  - Modelin eğitilmesi
  - Tahminlerin yapılması ve değerlendirilmesi
  - Hiperparametre optimizasyonu (GridSearch)  
**Yöntem:** Yukarıda belirtilen adımlar sırasıyla uygulanır ve nihai model değerlendirilir.

