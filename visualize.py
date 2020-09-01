import streamlit as st
from data import Data


def visualize():

    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("📊 visualisation des données")
    chart = st.selectbox("chart", ["bar", "line"])

    if chart == "bar":
        st.bar_chart(data.df.groupby([data.df.columns[-1]]).size())
    elif chart == "line":
        st.line_chart(data.df.groupby([data.df.columns[-1]]).size())