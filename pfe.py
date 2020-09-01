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
class Df:
    data = None
    filename = None

    x_test = None
    x_train = None
    y_test = None
    y_train = None

    classes = None

    # @st.cache(show_spinner=False)
    def loadData(self, filename):
        self.filename = filename
        data = pd.read_csv("/home/mehdi/pfe/app/data/" + filename)
        return data

    def initData(self, data):
        # data.columns = ["txId", "time step"] + [i for i in range(165)] + ["class"]
        self.data = data

    def showInfos(self):
        st.subheader("informations sur `%s`" % self.filename)
        st.write("number of rows :  `%d`" % self.data.shape[0])
        st.write("number of columns :  `%d`" % self.data.shape[1])
        st.write(st.table(self.data.groupby(["class"]).size()))
        # for i, v in self.data.groupby(["class"]).size().items():
        #     st.write(i, v)
        st.write(self.data.head())
        st.spinner()

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


# @st.cache(suppress_st_warning=True)
def file_selector(folder_path="data"):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    # return os.path.join(folder_path, selected_filename)
    return selected_filename


# uploader = st.empty()

# uploader.file_uploader("upload here")
# uploader.text("i am a text").balloons()
# uploader.balloons()
# uploader.table(pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"]))
# uploader.text_area("some label")
# uploader.text_input("input goes here", value="initial text")
# st.balloons()
# uploader.balloons()


def run():
    df = None
    res = st.sidebar.selectbox(
        "choose wisely", ["📥 upload", "🔍 explore", "📊 visualize", "🧮 classify"]
    )

    st.sidebar.button("hooo")
    if res == "📥 upload":
        # caching.clear_cache()
        filename = file_selector()
        st.write("You selected `%s`" % filename)
        if st.button("upload"):
            with st.spinner("Wait for it..."):
                df = Df()
                data = df.loadData(filename)
                df.initData(data)
            st.success("Done!")
    if res == "🔍 explore":
        # filename = file_selector()
        df = Df()
        # data = df.loadData(filename)
        # df.initData(data)
        df.showInfos()
    if res == "eat":
        st.write("you eat")

    # st.write("splitting data...")
    # df.split()
    # st.write("data splitted")

    # st.write("training data...")
    # cl = Classifier(df, 2)
    # cl.train()
    # st.write("data trained")

    # st.write("evaluating data...")
    # cl.evaluate()
    # st.write("data evaluated")


run()