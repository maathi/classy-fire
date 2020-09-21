import streamlit as st
from data import Data
import matplotlib.pyplot as plt
import seaborn as sns
def visualize():

    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("📊 visualisation des données")
    chart = st.selectbox("diagramme :", ["bar chart", "line chart"])

    x = st.selectbox("sur l'axe X :", data.df.columns)
    y = st.selectbox("sur l'axe Y :", data.df.columns)
   
    # licit = data.df[data.df['class'] == 1]
    # illicit = data.df[data.df['class'] == 0]

    # licit_x = licit[x].dropna(axis=0)
    # illicit_x = illicit[x].dropna(axis=0)

    # plt.hist(licit_x, bins=10, alpha=0.4, label='licites')
    # plt.hist(illicit_x, bins=10, alpha=0.4, label='illicites')
    # plt.legend(loc='upper right');
    # st.pyplot()

    # plt.pie([60,40],labels=['fo','fosfs'], autopct='%1.1f%%')
    # st.pyplot()

    atts = st.multiselect(
        'selectionnez les attributs',
        data.df.columns,
        [])

    if atts:
        corr_matrix = data.df[atts].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap='RdBu')
        st.pyplot()
        st.pyplot()
    # if chart == "bar chart":
    #     st.bar_chart(data.df.groupby([data.df.columns[-1]]).size())
    # elif chart == "line chart":
    #     st.line_chart(data.df.groupby([data.df.columns[-1]]).size())