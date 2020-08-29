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

    # @st.cache
    def load_data(self):
        self.edgelist = pd.read_csv("/home/mehdi/pfe/data/edgelist.csv", nrows=100)
        self.classes = pd.read_csv("/home/mehdi/pfe/data/classes.csv")
        self.features = pd.read_csv("/home/mehdi/pfe/data/features.csv", nrows=100)

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


data = Data()
st.write("loading data...")
data.load_data()
st.write("data loaded!")

st.write("cleaning data")
data.clean()
st.write("cleaned")
st.write(data.features)