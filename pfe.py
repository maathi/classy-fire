import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
import seaborn as sns


class Home:
    data = None
    x_test = None
    x_train = None
    y_test = None
    y_train = None
    model = None

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

    def train(self, zetaype):
        if zetaype == True:
            model = LogisticRegression().fit(self.x_train, self.y_train)
        else:
            model = RandomForestClassifier(
                n_estimators=50, max_depth=100, random_state=15
            ).fit(self.x_train, self.y_train)
        self.model = model

    def evaluate(self, model):
        preds = model.predict(self.x_test)
        cr = classification_report(self.y_test, preds)
        st.write(cr)


home = Home()
st.write("loading data...")
data = home.loadData()
home.initData(data)
st.write(data.head())
st.write("data loaded!")


st.write("splitting data...")
home.split()
st.write("data splitted")

st.write("training data...")
home.train(True)
st.write("data trained")

st.write("evaluating data...")
home.evaluate(home.model)
st.write("data evaluated")