import streamlit as st
import pandas as pd
import os
from data import Data
import upload
import infos
import visualize
import classify
import selection

def run():

    res = st.sidebar.selectbox(
        "", ["📤 upload", "🔍 informations", "📊 visualiser","selection", "🧮 classifier"]
    )

    if res == "📤 upload":
        upload.upload()
    if res == "🔍 informations":
        infos.infos()
    if res == "📊 visualiser":
        visualize.visualize()
    if res == "🧮 classifier":
        classify.classify()
    if res == "selection":
        selection.select()


run()