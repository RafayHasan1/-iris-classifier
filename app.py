

import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]
    return df, iris.target_names


@st.cache_resource
def train_model(df):
    X = df.drop(columns="species")
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


df, target_names = load_data()
model, accuracy = train_model(df)
features = df.columns[:-1]

st.title("🌸 Iris Flower Classifier")
st.write(
    "Move the sliders in the sidebar to describe a flower. "
    "The model will predict which of the three iris species it is."
)


st.sidebar.header("Flower measurements (cm)")
user_input = {}
for col in features:
    user_input[col] = st.sidebar.slider(
        label=col.replace(" (cm)", "").capitalize(),
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean()),
        step=0.1,
    )
input_df = pd.DataFrame([user_input])


prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

emoji = {"setosa": "🌼", "versicolor": "🌺", "virginica": "🌷"}
st.subheader("Prediction")
st.success(f"{emoji.get(prediction, '🌸')} This flower is most likely **Iris {prediction}**")

st.subheader("Confidence for each species")
proba_df = pd.DataFrame({"Probability": probabilities}, index=model.classes_)
st.bar_chart(proba_df)

with st.expander("Your input"):
    st.dataframe(input_df, hide_index=True)

with st.expander("About the model"):
    st.write(
        f"Algorithm: Random Forest (100 trees). "
        f"Trained on 120 flowers, tested on 30. Test accuracy: **{accuracy:.0%}**"
    )

with st.expander("See the dataset"):
    st.dataframe(df, hide_index=True)
    st.caption("150 flowers, 50 of each species (Fisher, 1936).")
