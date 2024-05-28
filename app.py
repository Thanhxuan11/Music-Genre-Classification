from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import joblib
import numpy as np
import librosa
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    low_energy = librosa.feature.rms(y=y)[0]

    features = np.array([
        np.mean(spectral_centroid), np.std(spectral_centroid),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flux), np.std(flux),
        np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
        np.mean(low_energy), np.std(low_energy)
    ])
    
    return features.reshape(1, -1)

def load_model_and_scaler():
    scaler_path = 'scaler.pkl'
    model_path = 'knn_model.pkl'
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model

scaler, model = load_model_and_scaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    audio_file = request.files['audioFile']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)
        features = extract_features(file_path)
        
        features_scaled = scaler.transform(features)
        predicted_label = model.predict(features_scaled)
        predicted_prob = model.predict_proba(features_scaled)
        
        return jsonify({
            'genre': predicted_label[0], 
            'probability': np.max(predicted_prob) * 100,
            'file_path': f'/uploads/{audio_file.filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
