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

    def init(self, filename):
        self.filename = filename
        self.df = pd.read_csv("/home/mehdi/pfe/app/data/" + filename)

    #     data.columns = ["txId", "time step"] + [i for i in range(165)] + ["class"]

    # def split(self):
    #     X = self.data[[i for i in range(165)]]
    #     Y = self.data["class"]
    #     x_train, x_test, y_train, y_test = train_test_split(
    #         X, Y, test_size=0.3, random_state=15, shuffle=False
    #     )
    #     self.x_test = x_test
    #     self.x_train = x_train
    #     self.y_test = y_test
    #     self.y_train = y_train


class Classifier:
    df = None
    clfType = None
    model = None

    def __init__(self, df, clfType):
        self.df = df
        self.clfType = clfType

    def train(self):
        if self.clfType == 1:
            model = LogisticRegression().fit(self.df.x_train, self.df.y_train)
        elif self.clfType == 2:
            model = RandomForestClassifier(
                n_estimators=50, max_depth=100, random_state=15
            ).fit(self.df.x_train, self.df.y_train)

        self.model = model

    def evaluate(self):
        preds = self.model.predict(self.df.x_test)
        cr = classification_report(self.df.y_test, preds)
        st.write(cr)
        st.balloons()


def file_selector(folder_path="data"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return selected_filename


def upload():
    filename = file_selector()
    st.write("You selected `%s`" % filename)
    if st.button("upload"):
        with st.spinner("Wait for it..."):
            data = Data()
            data.init(filename)
        st.success("Done!")


def infos():
    data = Data()
    st.subheader("informations sur `%s`" % data.filename)
    st.write("number of rows :  `%d`" % data.df.shape[0])
    st.write("number of columns :  `%d`" % data.df.shape[1])
    st.write(st.table(data.df.groupby(["class"]).size()))
    # for i, v in df.data.groupby(["class"]).size().items():
    #     st.write(i, v)
    st.write(data.df.head())


def visualize():
    data = Data()
    st.bar_chart(data.df.groupby(["class"]).size())


def classify():
    st.header("classifiying the data...")


def run():

    res = st.sidebar.selectbox(
        "choose wisely", ["📥 upload", "🔍 explore", "📊 visualize", "🧮 classify"]
    )

    if res == "📥 upload":
        upload()
    if res == "🔍 explore":
        infos()
    if res == "📊 visualize":
        visualize()
    if res == "🧮 classify":
        classify()


run()