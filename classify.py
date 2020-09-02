import streamlit as st
from data import Data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


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
        elif self.algo == "Decision Tree":
            model = DecisionTreeClassifier().fit(self.x_train, self.y_train)

        self.model = model

    def evaluate(self):
        preds = self.model.predict(self.x_test)
        cr = classification_report(self.y_test, preds)
        st.write(cr)
        st.balloons()


def classify():

    data = Data()
    if not data.loaded:
        st.error("vous devez charger des données d'abord!")
        return

    st.header("🧮 classification des données :")
    algo = st.selectbox(
        "sélectionnez l'algorithme de classification :",
        ["__", "Random Forest", "Logistic Regression", "Decision Tree"],
    )

    if algo != "__":
        with st.spinner("classification en cours..."):
            clf = Classifier(data, algo)
            clf.split()
            clf.train()
            clf.evaluate()