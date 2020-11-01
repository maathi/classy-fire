import streamlit as st
from data import Data

def cluster():
    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return
    st.header("🧿 Clustering :")

    test = st.selectbox(
        "sélectionnez l'algorithme de clustering :",
        ["__", "dbscan", "k-means"],
    )

    if test == "dbscan":
        eps = st.number_input("entrez la valeur du paramètre Ɛ :", 0.0, 5.0, 0.5, 0.1)
        nmin = st.number_input("entrez la valeur du paramètre n_min :", 1, 10, 5, 1)

        st.subheader("Résultat du clustering :")
        st.write("nombre de cluster : `%d`" % 1905)
        st.write("la table `%s` après clustering :" % "anons")
        st.write(data.df.head())
        st.button("sauvegarder")
        
    elif test == "k-means":
        folds = st.slider("nombre de plis :", 3, 20, 10, 1)