import numpy as np
import os
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load mô hình KNN và bộ chuẩn hóa từ các tệp pickle đã lưu trước đó
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Hàm trích xuất đặc trưng âm nhạc từ một file âm thanh
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    low_energy = librosa.feature.rms(y=y)[0]
    
    mean_spectral_centroid = np.mean(spectral_centroid)
    std_spectral_centroid = np.std(spectral_centroid)
    mean_rolloff = np.mean(rolloff)
    std_rolloff = np.std(rolloff)
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    mean_zero_crossing_rate = np.mean(zero_crossing_rate)
    std_zero_crossing_rate = np.std(zero_crossing_rate)
    mean_low_energy = np.mean(low_energy)
    std_low_energy = np.std(low_energy)
    
    feature_vector = np.array([mean_spectral_centroid, std_spectral_centroid,
                               mean_rolloff, std_rolloff,
                               mean_flux, std_flux,
                               mean_zero_crossing_rate, std_zero_crossing_rate,
                               mean_low_energy, std_low_energy])
    
    return feature_vector

# Mở bài hát và trích xuất đặc trưng
new_song_path = "D:\DA\VNTM3\Hatxam\Xam.085.wav"
new_song_features = extract_features(new_song_path)

# Chuẩn hóa đặc trưng của bài hát mới
new_song_features_scaled = scaler.transform(new_song_features.reshape(1, -1))

# Dự đoán thể loại của bài hát mới bằng mô hình KNN đã được tải
predicted_prob = knn_model.predict_proba(new_song_features_scaled)
predicted_label = knn_model.predict(new_song_features_scaled)

# Hiển thị kết quả dự đoán
print("Predicted genre:", predicted_label[0])
print("Confidence:", np.max(predicted_prob) * 100, "%")
