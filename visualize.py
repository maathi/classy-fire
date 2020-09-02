import streamlit as st
from data import Data


def visualize():

    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("📊 visualisation des données")
    chart = st.selectbox("diagramme :", ["bar chart", "line chart"])

    if chart == "bar chart":
        st.bar_chart(data.df.groupby([data.df.columns[-1]]).size())
    elif chart == "line chart":
        st.line_chart(data.df.groupby([data.df.columns[-1]]).size())