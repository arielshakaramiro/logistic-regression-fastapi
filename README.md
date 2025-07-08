# logistic-regression-fastapi
Repositori ini berisi project machine learning sederhana yang menggunakan Logistic Regression untuk klasifikasi biner berbasis nilai numerik, 
serta API untuk prediksi berbasis model tersebut menggunakan FastAPI. Project ini cocok untuk pembelajaran konsep supervised learning (klasifikasi), 
deployment model ML, dan pembuatan REST API sederhana.

## 🧠 Logistic Regression
Logistic Regression adalah algoritma Machine Learning yang digunakan untuk menyelesaikan masalah klasifikasi biner, yaitu ketika output hanya memiliki dua kemungkinan, seperti:\
  ✅ Ya / ❌ Tidak\
  📥 Spam / ✉️ Non-Spam\
  🎓 Lulus / ❌ Gagal\
Dalam proyek ini, logistic regression digunakan untuk memprediksi label 0 atau 1 berdasarkan satu nilai numerik input (X). Model dilatih menggunakan scikit-learn, 
dan disediakan API untuk prediksi berbasis FastAPI, sehingga model dapat digunakan secara real-time melalui HTTP request.

## 📁 Struktur Project
├── app.py                  # API untuk prediksi menggunakan FastAPI\
├── main.py                 # Training model logistic regression\
├── models/\
│   └── logistic_model.pkl  # Model Logistic Regression hasil training\
├── requirements.txt        # Daftar dependensi Python\
└── README.md               # Dokumentasi proyek

## 🧠 Penjelasan Singkat Kode
### main.py (Training Model)
- Membuat dataset dummy: X (fitur) dan Y (label 0 atau 1 berdasarkan threshold 5).
- Melatih model menggunakan LogisticRegression dari scikit-learn.
- Mengevaluasi akurasi model dan mencetak classification report.
- Menyimpan model ke file models/logistic_model.pkl menggunakan pickle.

### app.py (API Prediksi)
- Menggunakan FastAPI untuk menyediakan endpoint /predict.
- Model diload dari file logistic_model.pkl.
- Endpoint menerima input x (angka), lalu mengembalikan prediksi kelas (0 atau 1) dan probabilitasnya.

### requirements.txt
- Semua library yang dibutuhkan untuk menjalankan training model dan FastAPI server, termasuk scikit-learn, pandas, matplotlib, dan FastAPI.


## 🚀 Cara Menjalankan Proyek
### 1. Clone Repository
git clone https://github.com/username/logistic-regression-api.git \
cd logistic-regression-api 

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
python -m venv venv\
source venv/bin/activate  # Linux/Mac\
venv\Scripts\activate     # Windows

### 3. Install Dependensi
pip install -r requirements.txt

### 4. Jalankan Model Training
python main.py\
Model akan disimpan di folder models/logistic_model.pkl.

### 5. Jalankan FastAPI Server
uvicorn app:app --reload\
API akan aktif di: http://127.0.0.1:8000

### 6. Coba Endpoint /predict
Gunakan tools seperti Postman, Insomnia, atau langsung via terminal:\
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"x\": 6.5}"

Respons:\
{\
  "prediction": 1,\
  "probability": 0.97\
}

## 📷 Contoh Visualisasi
main.py juga menampilkan plot hasil prediksi model (aktual vs prediksi), menggunakan matplotlib.

## 🛠️ Tools & Library yang Digunakan
- Python 3.11+
- NumPy, Pandas, Matplotlib
- Scikit-learn
- FastAPI
- Uvicorn (server FastAPI)
- Pickle (penyimpanan model)

## 📚 Cocok Untuk
- Prediksi E-commerce
Memprediksi pembelian pelanggan berdasarkan faktor tertentu.
- Klasifikasi Email
Mengklasifikasikan email sebagai Spam atau Non-Spam.
- Prediksi Penyakit Jantung
Menilai risiko penyakit jantung berdasarkan riwayat kesehatan.

## Kesimpulan
Logistic Regression adalah algoritma yang sederhana dan efektif untuk 
masalah klasifikasi biner. Dengan memahami cara kerja algoritma ini 
serta cara melatih dan menyajikannya melalui API, kita dapat 
menggunakannya dalam berbagai aplikasi nyata.

