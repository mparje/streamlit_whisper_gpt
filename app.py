import streamlit as st
import pandas as pd
from audiorecorder import audiorecorder
from streamlit_player import st_player
from audio_recorder_streamlit import audio_recorder
import whisper
import torch
#torch.cuda.empty_cache()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

st.title('Pruebas de captura de voz usando audio recorder y transcripción usando whisper')

st.write("""Haga una grabación""")





model = whisper.load_model("base",'cpu')

audio_bytes = audio_recorder()



if audio_bytes:
    with open("audio.mp3", "wb") as file:
        file.write(audio_bytes)

    st.audio(audio_bytes, format="audio/wav")
    result = model.transcribe("audio.mp3",language='Spanish')
    #result = model.transcribe("audio.mp3",fp16=False, language='Spanish')
    st.text(result["text"])
