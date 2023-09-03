# Importing libraries
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pydub import AudioSegment
import pandas as pd
import speech_recognition as sr
import streamlit as st

def diarize(input):
    audio_file_path = sample_path + input
    audio_name = input.split('.')[0]
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
    # Perform diarization and dump the diarization output to disk using RTTM format
    diarization = pipeline(audio_file_path,num_speakers=2)
    output_path = audio_name + ".rttm"
    with open(output_path, "w") as rttm:
        diarization.write_rttm(rttm)    

    # Get dataframe
    def rttm_to_dataframe(rttm_file_path):
        columns = ["Type", "File ID","Channel","Start Time","Duration","Orthography","Confidence","Speaker","x","y"]
        with open(rttm_file_path,"r") as rttm_file:
            lines = rttm_file.readlines()
        data = []
        for line in lines:
            line = line.strip().split()
            data.append(line)
        df = pd.DataFrame(data, columns = columns)
        df = df.drop(["x","y","Orthography","Confidence"],axis=1)
        return df
    
    def extract_text_from_audio(audio_file_path, start_time, end_time):
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source, duration = end_time, offset = start_time)
        text = r.recognize_google(audio)
        return text
    rttm_file_path = output_path
    df = rttm_to_dataframe(rttm_file_path)
    df = df.astype({'Start Time':'float'})
    df = df.astype({'Duration':'float'})
    df['Utterance'] = None
    df['End Time'] =  df['Start Time'] + df['Duration']
    df.insert(1,'End Time',df.pop('End Time'))
    for ind in df.index:
        start_time = df['Start Time'][ind]
        end_time = df['End Time'][ind]
        try:
            transcription = extract_text_from_audio(audio_file_path, start_time, end_time)
            df['Utterance'][ind] = transcription
        except:
            df['Utterance'][ind] = 'Not Found'
    df = df.drop(["Type","File ID","Channel"],axis=1)
    return df


def button_click():
    st.header('Speaker diarization demo')
    inp = st.selectbox('Select an audio sample', ['Brooklyn 99', 'Gilmore Girls 01', 'Gilmore Girls 02', 'Home Alone', 'La La Land', 'Love and Other Drugs', 'New Girl 01', 'New Girl 02', 'Notebook', 'The Office 01', 'The Office 02'])
    if st.button('Go'):
        input_audio = audio_mapping[inp]
        st.write(inp)
        st.audio(sample_path+input_audio)
        output_df = diarize(input_audio)
        st.dataframe(output_df)


sample_path="wav/"
audio_mapping = {'Brooklyn 99':"brooklyn-sample.wav", 
                 'Gilmore Girls 01':"gilmore-01-sample.wav", 
                 'Gilmore Girls 02':"gilmore-02-sample.wav", 
                 'Home Alone':"homealone-sample.wav",
                 'La La Land':"lalaland-sample.wav", 
                 'Love and Other Drugs':"laod-sample.wav", 
                 'New Girl 01':"newgirl-01-sample.wav", 
                 'New Girl 02':"newgirl-02-sample.wav", 
                 'Notebook':"notebook-sample.wav", 
                 'The Office 01':"office-01-sample.wav", 
                 'The Office 02':"office-02-sample.wav"}


button_click()
