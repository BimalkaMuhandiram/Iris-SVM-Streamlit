import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import pickle

# Load the saved model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
col_name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=col_name)

# Streamlit app
st.title("Iris Classification with SVM")
st.sidebar.header("Dataset Overview and Visualization")

# Display dataset info
if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset Preview")
    st.write(dataset.head())

    st.subheader("Dataset Information")
    st.write(dataset.describe())

# Pair Plot
if st.sidebar.checkbox("Show Pair Plot"):
    st.subheader("Pair Plot")
    sns.set_palette('husl')
    pair_plot = sns.pairplot(dataset, hue='class', markers='+')
    st.pyplot(pair_plot.fig)

# Correlation Heatmap
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    numeric_data = dataset.select_dtypes(include=['number'])
    plt.figure(figsize=(7, 5))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='magma')
    st.pyplot()

# Violin Plots
if st.sidebar.checkbox("Show Violin Plots"):
    st.subheader("Violin Plots")
    for col in ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']:
        sns.violinplot(y='class', x=col, data=dataset, inner='quartile')
        st.pyplot()

# Model Evaluation
X = dataset.drop(['class'], axis=1)
y = dataset['class']
predictions = model.predict(X)

st.subheader("Model Evaluation")
accuracy = accuracy_score(y, predictions)
st.write(f"Accuracy: {accuracy:.2f}")

st.write("Classification Report:")
st.text(classification_report(y, predictions))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax)
st.pyplot(fig)
