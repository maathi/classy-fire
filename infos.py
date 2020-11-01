import streamlit as st
from data import Data
import pandas as pd
 


def infos():
    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return
    
    mem = data.df.memory_usage().sum() / 1048576 

    st.header("🔍 informations sur `%s` :" % data.filename)

    st.write("Nombre d'instances :  `%d`" % data.df.shape[0])
    atts = data.df.shape[1] - 1
    st.write("Nombre d'attributs :  `%d`" % atts)
    # st.write(
    #     "Nombre de classes :  `%d`"
    #     % data.df.groupby([data.df.columns[-1]]).size().count()
    # )
    st.write("espace mémoire occupé : `%0.2f` Mb" % mem)
    # st.write("les classes existantes et leurs fréquences correspondantes :")
    st.selectbox("sélectionnez un attribut :",["class"])
    st.write("les valeurs de l'attributs `%s` et leurs fréquences :" % "class")
    # st.table(data.df.groupby([data.df.columns[-1]]).size())
    fata = [['licite', 126347], ['illicite', 77421]] 
  
    # Create the pandas DataFrame 
    fata = pd.DataFrame(fata, columns = ['valeur', 'fréquence']) 
    st.table(fata)
    st.write("statistiques descriptives :")
    st.write(data.df.describe())
    # for i, v in df.data.groupby(["class"]).size().items():
    #     st.write(i, v)
    # if st.checkbox("afficher la table %s :" % data.filename[:-4]):
    st.write("affichage de la table `%s` :" % data.filename[:-4])
    n_rows = st.slider("sélectionnez le nombre de lignes à afficher :", 5, 20, 10)
    st.write(data.df.head(n_rows))