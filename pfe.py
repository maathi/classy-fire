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


class Data:
    features = None
    classes = None
    edgelist = None
    x_test = None
    x_train = None
    y_test = None
    y_train = None

    @st.cache
    def load_data(self):
        edgelist = pd.read_csv("/home/mehdi/pfe/data/edgelist.csv")
        classes = pd.read_csv("/home/mehdi/pfe/data/classes.csv")
        features = pd.read_csv("/home/mehdi/pfe/data/features.csv")

        return edgelist, classes, features

    def initvars(self, edgelist, classes, features):
        self.edgelist = edgelist
        self.features = features
        self.classes = classes

    def clean(self):
        self.features.columns = ["txId", "time step"] + [i for i in range(165)]
        # fusionner les table 'classes' et 'futures'
        self.features = pd.merge(
            self.features, self.classes, left_on="txId", right_on="txId", how="left"
        )
        # remplacer 'unknown' par '0'
        self.features["class"] = self.features["class"].apply(
            lambda x: "0" if x == "unknown" else x
        )

    def split(self):
        data = self.features[self.features["class"] != "0"]
        X = data[[i for i in range(165)]]
        Y = data["class"]
        # Y = Y.apply(lambda x: 0 if x == '2' else 1 )
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


data = Data()
st.write("loading data...")
edgelist, classes, features = data.load_data()
data.initvars(edgelist, classes, features)
st.write("data loaded!")

st.write("cleaning data...")
data.clean()
st.write("cleaned")
st.write(data.features.head())


st.write("splitting data...")
data.split()
st.write("data splitted")

st.write("training data...")
data.train(False)
st.write("data trained")

st.write("evaluating data...")
data.evaluate(data.model)
st.write("data evaluated")