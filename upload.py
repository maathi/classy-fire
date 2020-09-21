import streamlit as st
import os
from data import Data
import pandas as pd

def file_selector(folder_path="data"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("sélectionnez votre dataset :", filenames)
    return selected_filename


def upload():
    st.header("📤 upload")
    
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    # uploaded_file = st.file_uploader("sélectionnez votre dataset :", type="csv")
    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file)

    filename = file_selector()
    if st.button("upload"):
        with st.spinner("chargement en cours..."):
            data = Data()
            data.init(filename)
        st.success("le dataset a été chargé!")