import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import time
from PIL import Image
import os
from streamlit import caching


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
class Data:
    filename = None
    df = None
    loaded = False

    def init(self, filename):
        self.filename = filename
        self.df = pd.read_csv("/home/mehdi/pfe/app/data/" + filename)
        self.loaded = True

    #     data.columns = ["txId", "time step"] + [i for i in range(165)] + ["class"]


class Classifier:
    df = None
    algo = None
    model = None

    def __init__(self, data, algo):
        self.df = data.df
        self.algo = algo

    def split(self):
        X = self.df[self.df.columns[:-1]]
        Y = self.df[self.df.columns[-1:]]

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=15, shuffle=False
        )
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train

    def train(self):
        if self.algo == "Logistic Regression":
            model = LogisticRegression().fit(self.x_train, self.y_train)
        elif self.algo == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=50, max_depth=100, random_state=15
            ).fit(self.x_train, self.y_train)

        self.model = model

    def evaluate(self):
        preds = self.model.predict(self.x_test)
        cr = classification_report(self.y_test, preds)
        st.write(cr)
        st.balloons()


def file_selector(folder_path="data"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return selected_filename


##################
#########################
#########
#######################################3


def upload():
    st.header("📤 upload")

    filename = file_selector()
    st.write("You selected `%s`" % filename)
    if st.button("upload"):
        with st.spinner("Wait for it..."):
            data = Data()
            data.init(filename)
        st.success("Done!")


def infos():
    st.header("🔍 informations")
    data = Data()
    if not data.loaded:
        st.write("upload first")
        return

    st.header("informations sur `%s` :" % data.filename)

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


def visualize():
    st.header("📊 visualiser")
    data = Data()
    if not data.loaded:
        st.write("upload first")
        return

    chart = st.selectbox("chart", ["bar", "line"])

    if chart == "bar":
        st.bar_chart(data.df.groupby([data.df.columns[-1]]).size())
    elif chart == "line":
        st.line_chart(data.df.groupby([data.df.columns[-1]]).size())


def classify():
    st.header("🧮 classification des données :")
    data = Data()
    if not data.loaded:
        st.write("upload first")
        return

    algo = st.selectbox(
        "sélectionnez l'algorithme de classification :",
        ["__","Random Forest", "Logistic Regression"],
    )
    
    if algo != "__":
        with st.spinner("classification en cours..."):
            clf = Classifier(data, algo)
            clf.split()
            clf.train()
            clf.evaluate()

def run():

    res = st.sidebar.selectbox(
        "choose wisely", ["📤 upload", "🔍 informations", "📊 visualiser", "🧮 classifier"]
    )

    if res == "📤 upload":
        upload()
    if res == "🔍 informations":
        infos()
    if res == "📊 visualiser":
        visualize()
    if res == "🧮 classifier":
        classify()


run()