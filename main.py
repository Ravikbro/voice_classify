import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image, ImageTk


duration = 5  
sample_rate = 44100  
model_path = 'saved_models/audio_classification.hdf5'
metadata_path = 'data/metadata/UrbanSound8K.csv'

def start_recording():
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait for the recording to complete
    return recording

def save_audio(recording):
    file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Files", "*.wav")])
    if file_path:
        sf.write(file_path, recording, sample_rate)
        print("Audio saved as:", file_path)
        return file_path
    return None

def predict_audio(file_path):
    model = load_model(model_path)
    audio, sr = librosa.load(file_path)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    mfcc_scaled_features = mfcc_scaled_features.reshape(1, -1)
    predicted_probabilities = model.predict(mfcc_scaled_features)
    predicted_label_index = np.argmax(predicted_probabilities)

    df = pd.read_csv(metadata_path)
    predicted_labels = df['class'].unique().tolist()
    predicted_labels.sort()
    predicted_label = predicted_labels[predicted_label_index]
    print('Predicted Label:', predicted_label)
    predicted_label_text.set('Predicted Label: ' + predicted_label)

def record_and_predict():
    recording = start_recording()
    file_path = save_audio(recording)
    if file_path:
        predict_audio(file_path)

root = tk.Tk()
root.title("Voice Recognition Model")
root.geometry("600x300")

background_image = Image.open("download.jpeg")  # Replace with the path to your image
background_image = background_image.resize((600, 300))  # Set the desired size
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

predicted_label_text = tk.StringVar()  # Variable to store the predicted label

def open_record_window():
    record_window = tk.Toplevel(root)
    record_window.title("Record and Predict")
    record_window.geometry("300x150")

    record_button = tk.Button(record_window, text="Record and Predict", command=record_and_predict)
    record_button.pack(pady=10)

    predicted_label_label = tk.Label(record_window, textvariable=predicted_label_text)
    predicted_label_label.pack()

main_label = tk.Label(root, text="Welcome to Audio Recorder")
main_label.pack(pady=20)

open_record_button = tk.Button(root, text="Open Recorder", command=open_record_window)
open_record_button.pack()

root.mainloop()
