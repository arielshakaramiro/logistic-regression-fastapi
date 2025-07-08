# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Membuat dataset sederhana
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Variabel independen (X)
Y = (
    (X > 5).astype(int).ravel()
)  # Label biner (0 atau 1), jika X > 5 maka 1, jika tidak maka 0

# Menampilkan beberapa sampel data
print("Sampel Data:")
print(pd.DataFrame(np.hstack((X, Y.reshape(-1, 1))), columns=["X", "Y"]).head())

# 2. Membagi dataset menjadi training dan testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Membuat model Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# 4. Melihat hasil model
print("\nHasil Training Model:")
print(f"Intercept (b0): {model.intercept_[0]:.2f}")
print(f"Koefisien (b1): {model.coef_[0][0]:.2f}")

# 5. Prediksi pada data uji
Y_pred = model.predict(X_test)

# 6. Evaluasi model
accuracy = accuracy_score(Y_test, Y_pred)
print("\nEvaluasi Model:")
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(Y_test, Y_pred))

# 7. Visualisasi hasil klasifikasi
plt.scatter(X_test, Y_test, color="blue", label="Actual Data")
plt.scatter(X_test, Y_pred, color="red", marker="x", label="Predicted")
plt.xlabel("X")
plt.ylabel("Class")
plt.legend()
plt.title("Hasil Logistic Regression")
plt.show()

plt.savefig("outputs/hasil_logistic_regression.png")  # simpan gambar

# Simpan model ke dalam file menggunakan pickle
import pickle

with open("models/logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil disimpan sebagai 'logistic_model.pkl'")
