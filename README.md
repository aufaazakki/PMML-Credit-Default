# PMML-Credit-Default
# UTS Prediksi Modern & Machine Learning: Credit Default Prediction

Eksperimen machine learning untuk memprediksi apakah nasabah kartu kredit akan mengalami **default pembayaran pada bulan berikutnya**, menggunakan dataset *Default of Credit Card Clients* (UCI/Kaggle).

**Nama:** Aufaa Zakki Maulidan
**NIM:** 2404220034
**Rombel:** 1
**Dosen Pengampu:** Nur Achmey Selgi Harwanti, S.Stat., M.Stat.
**Prodi:** Sarjana Statistika dan Sains Data, FMIPA, Universitas Negeri Semarang

---

## Deskripsi Singkat

Proyek ini berbentuk *Experiment Challenge*: model dibangun secara bertahap untuk melihat bagaimana preprocessing, pemilihan model, hyperparameter tuning, dan model advanced memengaruhi performa. Metrik utama adalah **F1-score**, dengan Accuracy, Precision, dan Recall sebagai metrik pendukung.

## Dataset

- Sumber: [Default of Credit Card Clients Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- Jumlah data: 30.000 observasi, 23 predictor (setelah `ID` dikeluarkan)
- Target: `default.payment.next.month` (0 = tidak default, 1 = default)
- Kelas tidak seimbang: 77,88% tidak default dan 22,12% default

## Isi Repository

| File | Keterangan |
|---|---|
| `UTS_PMML_Credit_Default.ipynb` | Notebook utama (Experiment 0 sampai 5, lengkap dengan kode dan output) |
| `UCI_Credit_Card.csv` | Dataset yang dipakai |

## Alur Eksperimen

| Experiment | Isi |
|---|---|
| 0 | Data understanding & preparation: cek missing value, duplicate, nilai tidak wajar, distribusi target, train-test split 80:20 (stratified) |
| 1 | Baseline: Logistic Regression |
| 2 | Preprocessing: StandardScaler + OneHotEncoder di dalam `Pipeline` |
| 3 | Perbandingan model: Logistic Regression, Decision Tree, Random Forest |
| 4 | Hyperparameter tuning Random Forest dengan `GridSearchCV` (5-fold CV) |
| 5 | Advanced model: XGBoost |

Seluruh evaluasi memakai 5-fold Stratified Cross Validation pada training set. Preprocessing dilakukan di dalam `Pipeline` dan split dilakukan sebelum training, sehingga tidak terjadi data leakage.

## Ringkasan Hasil

| Experiment | Model | CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|---|
| 1 | Baseline (Logistic Regression) | 0,3367 | 0,8082 | 0,6648 | 0,2675 | 0,3815 |
| 2 | Baseline + Preprocessing | 0,3648 | 0,8090 | 0,6930 | 0,2449 | 0,3619 |
| 3 | Random Forest (terbaik di perbandingan model) | 0,4769 | 0,8163 | 0,6502 | 0,3670 | 0,4692 |
| 4 | Random Forest (tuned) | 0,5453 | 0,7907 | 0,5248 | 0,5652 | 0,5443 |
| 5 | XGBoost (scale_pos_weight) | 0,5420 | 0,7607 | 0,4687 | 0,6149 | 0,5319 |

## Model Final

**Random Forest (tuned)** dengan `n_estimators=200`, `max_depth=10`, `min_samples_leaf=2`, dan `class_weight='balanced'`.

Model ini dipilih karena memiliki CV F1 dan Test F1 tertinggi, lebih sederhana daripada XGBoost, dan recall-nya naik dari 0,2675 (baseline) menjadi 0,5652.

## Temuan Utama

- Preprocessing (scaling dan encoding) meningkatkan stabilitas dan CV F1, tetapi tidak meningkatkan Test F1.
- Hyperparameter tuning meningkatkan Test F1 dari 0,4692 menjadi 0,5443, terutama lewat `class_weight='balanced'` dan `max_depth=10`.
- Model yang lebih kompleks (XGBoost) tidak lebih baik daripada Random Forest tuned.
- Peningkatan performa lebih banyak datang dari penanganan ketidakseimbangan kelas daripada kompleksitas algoritma.

## Cara Menjalankan

1. Buka `UTS_PMML_Credit_Default.ipynb` di Google Colab.
2. Upload `UCI_Credit_Card.csv` ke environment Colab (satu folder dengan notebook).
3. Jalankan semua sel secara berurutan (*Runtime → Run all*). Library yang dipakai: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost.
