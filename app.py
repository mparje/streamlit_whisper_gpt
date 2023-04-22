import streamlit as st
import pandas as pd
from audiorecorder import audiorecorder
from streamlit_player import st_player
from audio_recorder_streamlit import audio_recorder
import whisper
import io

st.title('Pruebas de captura de voz usando audio recorder y transcripción usando whisper')

st.write("""Haga una grabación""")

# Load the Whisper model
model = whisper.Transcriber(model_path='base')

# Record audio
audio_bytes = audio_recorder()

if audio_bytes:
    # Write audio to file
    with open('audio.mp3', 'wb') as f:
        f.write(audio_bytes)

    # Transcribe audio
    with open('audio.mp3', 'rb') as f:
        audio_bytes = io.BytesIO(f.read())
    result = model.transcribe(audio_bytes, language='Spanish')

    # Display audio and transcription
    st.audio(audio_bytes, format='audio/wav')
    st.text(result['text'])
