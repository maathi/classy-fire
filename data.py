import streamlit as st
import pandas as pd


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
class Data:
    filename = None
    df = None
    loaded = False

    def init(self, filename):
        self.filename = filename
        self.df = pd.read_csv("data/" + filename)
        self.loaded = True
