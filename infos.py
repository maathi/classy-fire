import streamlit as st
from data import Data
import pandas as pd


def infos():
    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return
    st.write(type(data.df.info()))
    st.write(data.df.describe())
    st.header("🔍 informations sur `%s` :" % data.filename)
    st.write("Nombre de lignes :  `%d`" % data.df.shape[0])
    st.write("Nombre de colonnes :  `%d`" % data.df.shape[1])
    st.write(
        "Nombre de classes :  `%d`"
        % data.df.groupby([data.df.columns[-1]]).size().count()
    )
    st.write("les classes existantes et leurs fréquences correspondantes :")
    st.write(st.table(data.df.groupby([data.df.columns[-1]]).size()))

    # for i, v in df.data.groupby(["class"]).size().items():
    #     st.write(i, v)
    st.subheader("affichage de la table `%s` :" % data.filename[:-4])
    n_rows = st.slider("sélectionnez le nombre de lignes à afficher :", 5, 20, 10)
    st.write(data.df.head(n_rows))