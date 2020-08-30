import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import os


class Df:
    data = None
    x_test = None
    x_train = None
    y_test = None
    y_train = None

    @st.cache
    def loadData(self):
        data = pd.read_csv("/home/mehdi/pfe/data/data.csv")
        return data

    def initData(self, data):
        data.columns = ["txId", "time step"] + [i for i in range(165)] + ["class"]
        self.data = data

    def split(self):
        X = self.data[[i for i in range(165)]]
        Y = self.data["class"]
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=15, shuffle=False
        )
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train


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


# def file_selector(folder_path="."):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox("Select a file", filenames)
#     return os.path.join(folder_path, selected_filename)


# uploader = st.empty()
# filename = file_selector()
# uploader.file_uploader("upload here")
# uploader.text("i am a text")
# uploader.table(pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"]))
# uploader.text_area("some label")
# uploader.text_input("input goes here", value="initial text")
# st.balloons()
# uploader.balloons()
# st.write('You selected `%s`' % filename)


def run():
    df = Df()
    st.write("loading data...")
    data = df.loadData()
    df.initData(data)
    st.write(data.head())
    st.write("data loaded!")

    st.write("splitting data...")
    df.split()
    st.write("data splitted")

    st.write("training data...")
    cl = Classifier(df, 2)
    cl.train()
    st.write("data trained")

    st.write("evaluating data...")
    cl.evaluate()
    st.write("data evaluated")


run()