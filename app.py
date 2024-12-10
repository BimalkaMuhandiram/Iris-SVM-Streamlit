import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle

# Title
st.title("Iris Classification Model Performance")
st.write("This app demonstrates the performance of the SVM model created for the Iris dataset.")

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=col_names)

# Display dataset
st.subheader("Iris Dataset")
st.write("The dataset used to train the model is displayed below:")
st.dataframe(dataset)

# Load the trained model
@st.cache
def load_model():
    with open('SVM.pickle', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sidebar input for user test data
st.sidebar.header("Input Test Data")
sepal_length = st.sidebar.slider('Sepal Length', float(dataset['sepal-length'].min()), float(dataset['sepal-length'].max()), float(dataset['sepal-length'].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(dataset['sepal-width'].min()), float(dataset['sepal-width'].max()), float(dataset['sepal-width'].mean()))
petal_length = st.sidebar.slider('Petal Length', float(dataset['petal-length'].min()), float(dataset['petal-length'].max()), float(dataset['petal-length'].mean()))
petal_width = st.sidebar.slider('Petal Width', float(dataset['petal-width'].min()), float(dataset['petal-width'].max()), float(dataset['petal-width'].mean()))

# Prediction for user input
input_features = pd.DataFrame({
    'sepal-length': [sepal_length],
    'sepal-width': [sepal_width],
    'petal-length': [petal_length],
    'petal-width': [petal_width]
})
prediction = model.predict(input_features)

st.subheader("Model Prediction for Input")
st.write(f"The predicted class for the given input is: **{prediction[0]}**")

# Split the dataset for evaluation
from sklearn.model_selection import train_test_split
X = dataset.drop(['class'], axis=1)
y = dataset['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model Predictions
test_predictions = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, test_predictions)
classification = classification_report(y_test, test_predictions, output_dict=True)

# Display Accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy on the test set: **{accuracy:.2f}**")

# Classification Report
st.subheader("Classification Report")
st.dataframe(pd.DataFrame(classification).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, test_predictions, labels=model.classes_)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax)
st.pyplot(fig)

# Pair Plot Visualization
st.subheader("Pair Plot of Dataset")
st.write("A pair plot to visualize the features of the Iris dataset:")
pair_plot = sns.pairplot(dataset, hue='class', markers='+')
st.pyplot(pair_plot)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
st.write("Correlation heatmap showing relationships between numeric features:")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(dataset.corr(), annot=True, cmap='magma', ax=ax)
st.pyplot(fig)