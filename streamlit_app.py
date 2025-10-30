import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

csv_file = "EDA_Formatted_Data.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.success(" Dataset loaded successfully!")
else:
    st.error(f"❌ {csv_file} not found!")

# Train the model directly in the app
if 'df' in locals():
    X = df[['Internal Marks (Standardized)', 'Preboard Marks (Standardized)']]
    y = df['Predicted Grade']
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    st.success(" Model trained successfully!")
else:
    st.error("Dataset not loaded, cannot train model.")

st.title("Student Grade Prediction App (KNN)")

st.header("Predict Student Grade")
internal_marks = st.number_input("Internal Marks (Standardized)", 0.0, 1.0, step=0.01)
preboard_marks = st.number_input("Preboard Marks (Standardized)", 0.0, 1.0, step=0.01)

if st.button("Predict Grade"):
    if 'knn' in locals():
        prediction = knn.predict([[internal_marks, preboard_marks]])
        st.success(f"Predicted Grade: {prediction[0]}")

if 'df' in locals():
    st.header("Data Visualizations")
    st.subheader("Grade Distribution")
    st.bar_chart(df['Predicted Grade'].value_counts())
    st.subheader("Internal vs Preboard Marks")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        x='Internal Marks (Standardized)',
        y='Preboard Marks (Standardized)',
        hue='Predicted Grade',
        data=df, ax=ax1
    )
    st.pyplot(fig1)
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
