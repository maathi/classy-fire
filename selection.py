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
    st.selectbox("", ["corr"])
    if st.button("appliquer"):
        corr_matrix = data.df.corr()
        # plt.figure(figsize=(8,6))
        # plt.title('Correlation Heatmap of Iris Dataset')
        # a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
        # a.set_xticklabels(a.get_xticklabels(), rotation=30)
        # a.set_yticklabels(a.get_yticklabels(), rotation=30)           
        # plt.show()    
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    
    
        # Find index of feature columns with correlation greater than 0.9
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        st.write(len(to_drop))
        st.write(to_drop)
        data.df = data.df.drop(to_drop, axis=1)
        st.write(data.df.head())
    