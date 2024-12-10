import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix  # <-- added confusion_matrix import
import pickle
import time

# Load the saved SVM model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the Iris dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
col_name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=col_name)

# Sidebar: Model controls and options
st.sidebar.title("ML Model Controls")
st.sidebar.header("Upload Image and Model Settings")

# File uploader for image (useful if you want to later integrate image classification)
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# Slider for setting test size
test_size = st.sidebar.slider("Select Test Size", 0.1, 0.9, 0.2)

# Main container for displaying content
container = st.container()

# Header for the app
st.title("Iris Classification with SVM")
container.header("Explore and Analyze the Iris Dataset")

# Display Dataset Information
if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset Preview")
    st.write(dataset.head())

    st.subheader("Dataset Description")
    st.write(dataset.describe())

# Pair Plot Visualization
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

# Display uploaded image (if any)
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

# Model Training and Evaluation
st.subheader("Model Evaluation")

# Prepare the data for SVM model
X = dataset.drop(['class'], axis=1)
y = dataset['class']

# Show progress bar for model training
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("Training model...")
time.sleep(2)  # Simulate time for model training
progress_bar.progress(50)
status_text.text("Evaluating model...")
time.sleep(2)  # Simulate time for evaluation
progress_bar.progress(100)

# Make predictions using the pre-trained model
predictions = model.predict(X)

# Accuracy and Classification Report
accuracy = accuracy_score(y, predictions)
st.write(f"Accuracy: {accuracy:.2f}")

st.write("Classification Report:")
st.text(classification_report(y, predictions))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y, predictions)  # <-- Now works as confusion_matrix is imported
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(ax=ax)
st.pyplot(fig)

# Sidebar for input widgets (if any additional inputs are needed)
st.sidebar.header("Adjust Model Parameters")
# Example: Adding a slider for model parameters (e.g., SVM kernel, C parameter)
kernel = st.sidebar.selectbox("Choose Kernel", ["linear", "rbf", "poly", "sigmoid"])
C_value = st.sidebar.slider("Select C Value (Regularization)", 0.1, 10.0, 1.0)

# Button to trigger predictions (example for future input-based predictions)
if st.sidebar.button("Make Predictions"):
    sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example sample (sepal-length, sepal-width, petal-length, petal-width)
    prediction = model.predict(sample_data)
    st.write(f"Prediction for the sample data: {prediction[0]}")

# Displaying some additional graphs
st.subheader("Feature Importance or Other Graphs")
sns.barplot(x=dataset.columns[:-1], y=model.coef_.flatten())
plt.title("Feature Importance")
st.pyplot()
