import streamlit as st
import pandas as pd
import os
from data import Data
import upload
import infos
import visualize
import classify
import selection
import clustering

def run():

    res = st.sidebar.selectbox(
        "", ["📤 Upload", "🔍 Informations", "📊 Visualisation","🎚 Sélection","🧿 Clustering", "🧮 Classification"]
    )

    if res == "📤 Upload":
        upload.upload()
    if res == "🔍 Informations":
        infos.infos()
    if res == "📊 Visualisation":
        visualize.visualize()
    if res == "🎚 Sélection":
        selection.select()
    if res == "🧿 Clustering":
        clustering.cluster()
    if res == "🧮 Classification":
        classify.classify()



run()