import streamlit as st
import os
from data import Data


def file_selector(folder_path="data"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return selected_filename


def upload():
    st.header("📤 upload")

    filename = file_selector()
    st.write("You selected `%s`" % filename)
    if st.button("upload"):
        with st.spinner("Wait for it..."):
            data = Data()
            data.init(filename)
        st.success("Done!")