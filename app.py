import streamlit as st
import pandas as pd
import sounddevice as sd
import io
import scipy.io.wavfile as wav

import whisper

st.title('Pruebas de captura de voz usando audio recorder y transcripción usando whisper')

st.write("""Haga una grabación""")

# Record audio
duration = 10  # seconds
fs = 44100  # sampling rate
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
wav_bytes = io.BytesIO()
wav.write(wav_bytes, fs, audio)

# Transcribe audio
model = whisper.Transcriber(model_path='base')
result = model.transcribe(wav_bytes.getvalue(), language='es-MX')

# Display audio and transcription
st.audio(wav_bytes.getvalue(), format='audio/wav')
st.text(result['text'])
