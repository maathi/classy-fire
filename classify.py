import streamlit as st
import pandas as pd
import numpy as np
from data import Data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import time

class Classifier:
    df = None
    algo = None
    model = None

    def __init__(self, data, algo):
        self.df = data.df
        self.algo = algo

        self.X = self.df[self.df.columns[:-1]]
        self.Y = self.df[self.df.columns[-1:]]

    def setModel(self):
        if self.algo == "Logistic Regression":
            model = LogisticRegression()
        elif self.algo == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=100, random_state=15
                )
        elif self.algo == "Decision Tree":
            model = DecisionTreeClassifier()
        elif self.algo == "svm":
            model = svm.SVC()

        self.model = model

    def report(self, y_test, y_pred, d):
        #replace this with heatmap thing
        st.subheader("Rapport de classification :")
        st.write("matrice de confusion :")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        
        aa = accuracy_score(y_test, y_pred)
        pp = precision_score(y_test, y_pred)
        rr = recall_score(y_test, y_pred)
        
        st.write("Taux de succès : `%0.2f`" % aa)
        st.write("Précision : `%0.2f`" % pp)
        st.write("Rappel :`%0.2f`" % rr)
        st.write("time taken to build model : `%0.2f`s" % d)

        st.balloons()


class Classifier_t(Classifier):
    pass

    def split(self, size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=size, random_state=15, shuffle=False
        )

    def train(self):
        t0 = time.time()
        self.model = self.model.fit(self.x_train, self.y_train)
        self.preds = self.model.predict(self.x_test)
        self.d = round(time.time() - t0, 3)


    def evaluate(self):
        self.report(self.y_test, self.preds, self.d)
 



class Classifier_c(Classifier):
    pass

    def train(self, folds):    
        t0 = time.time()
        self.y_pred = cross_val_predict(self.model, self.X, self.Y, cv=folds)
        self.d = round(time.time() - t0, 3)

    def evaluate(self):
        self.report(self.Y, self.y_pred, self.d)


def classify():

    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("🧮 classification des données :")

    test = st.selectbox(
        "sélectionnez la Méthode de test :",
        ["__", "Percentage split", "Cross-validation"],
    )

    if test == "Percentage split":
        size = st.slider("selection",0.0,1.0,0.25,0.05)
    elif test == "Cross-validation":
        folds = st.slider("plis :", 3, 20, 10, 1)

    algo = st.selectbox(
        "sélectionnez l'algorithme de classification :",
        ["__", "Random Forest", "Logistic Regression", "Decision Tree","svm"],
    )

    
    if algo != "__" and test != "__":
        with st.spinner("classification en cours..."):
            if test == "Percentage split":
                clf = Classifier_t(data, algo)
                clf.split(size)
                clf.setModel()
                clf.train()
                clf.evaluate()
            elif test == "Cross-validation":
                clf = Classifier_c(data, algo)
                clf.setModel()
                clf.train(folds)
                clf.evaluate()
