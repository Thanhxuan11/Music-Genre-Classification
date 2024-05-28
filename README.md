Description:

This project implements a web application using Flask to classify audio files into different genres. Users can upload audio files, and the application predicts the genre with the highest probability.

Features:

Uploads audio files.
Extracts Mel-frequency cepstral coefficients (MFCCs) and other audio features.
Classifies audio using a pre-trained K-Nearest Neighbors (KNN) model.
Displays the predicted genre and probability.
Requirements:

Python 3.x
Flask
librosa
joblib
NumPy (numpy)
A pre-trained KNN model file (knn_model.pkl)
A scaler file (scaler.pkl) - This file likely stores parameters used for normalization during feature extraction.
Installation:

Clone this repository.
Install the required libraries using pip install Flask librosa joblib numpy.
Download the pre-trained KNN model (knn_model.pkl) and scaler file (scaler.pkl) and place them in the project directory.
Usage:

Run the application using python app.py.
Open http://127.0.0.1:5000/ (or your local development server address) in your web browser.
Click "Choose File" and select an audio file.
Click "Upload."
The application will predict the genre and probability and display the results on the webpage.
Code Structure:

app.py: The main Flask application file containing routes and logic for handling file uploads, feature extraction, model prediction, and response generation.
uploads: This folder stores uploaded audio files.
extract_features.py: This file defines functions for extracting audio features like spectral centroid, rolloff, flux, zero-crossing rate, and low energy from audio files using librosa.
load_model_and_scaler.py: This file loads the pre-trained KNN model and scaler using joblib.
Additional Notes:

This is a basic example of audio genre classification using Flask. You can further enhance it by:
Supporting more audio formats.
Implementing functionalities for pre-processing audio (e.g., noise reduction).
Integrating with a database to store predictions.
Providing a more user-friendly interface.
Ensure you have the pre-trained KNN model and scaler files for the application to function correctly.

![image](https://github.com/Thanhxuan11/Music-Genre-Classification/assets/117796081/2f3c630d-715b-4854-9d57-41f14820a1b6)
