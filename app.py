from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_filename = 'modelForPrediction1 (1).sav'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)

# @app.route('/')
# def home():
#     return render_template("Home.html")

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/emotion')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded audio file
        audio_file = request.files['file']
        
        # Load the audio file using librosa
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract features (ensure these match how your model was trained)
        feature = extract_feature(y, sr, mfcc=True, chroma=True, mel=True)
        feature = feature.reshape(1, -1)
        
        # Predict emotion
        prediction = model.predict(feature)[0]
        print(prediction)
        return render_template('index.html', emotion=prediction)
    except Exception as e:
        return str(e), 400

def extract_feature(y, sr, mfcc=True, chroma=True, mel=True):
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
    return result

if __name__ == '__main__':
    app.run(debug=True)
