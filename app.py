import pickle
import string
import streamlit as st
import numpy as np

global Lrdetect_Model

Lrdetect_File = open('model.pckl','rb')
Lrdetect_Model = pickle.load(Lrdetect_File)
Lrdetect_File.close()

def make_prediction(text):
    text = np.array_str(text)
    for i in string.punctuation:
        text = text.replace(i,'')
    return text


st.title("Language Detection Tool")
st.header("This language detection tool can detect 17 different languages!")
st.subheader("Including English, Hindi, German, Arabic, Russian...")
input_text = st.text_input("Provide your text input here")
button_clicked = st.button("Detect Language")
if button_clicked:
    output = make_prediction((Lrdetect_Model.predict([input_text])))
    st.write("The input text is in ",output,".")
