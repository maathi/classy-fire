import streamlit as st 
import numpy as np
from data import Data
import classify

def select():
    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("🎚 sélection d'attributs :")
    meth = st.selectbox("choisissez la méthode de sélection :", ["__", "matrice de corrélation"])
    
    if meth == "matrice de corrélation":
        min_corr = st.slider("sélectionnez la corrélation minimum", 0.5, 1.0, 0.9, 0.01)
        corr_matrix = data.df.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    
    
        # Find index of feature columns with correlation greater than 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > min_corr)]
 
        st.subheader("Résultat de la sélection :")
        st.write("le nombre d'attributs supprimés : `%d`" % len(to_drop))
        st.write("la liste des attributs supprimés :")
        st.write(to_drop)
        data.df = data.df.drop(to_drop, axis=1)
        
    