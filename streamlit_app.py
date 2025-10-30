import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Sample data for demonstration (replace with your actual data upload or URL)
data = {
    'Internal Marks (Standardized)': [0.136401839, -1.191866066, -1.002113509, -0.432855835, -1.381618624],
    'Preboard Marks (Standardized)': [0.642281098, 0.884540915, 0.157761465, -0.972784346, -0.165251624],
    'Predicted Grade': [8, 8, 7, 6, 6]
}
df = pd.DataFrame(data)

# Train the model directly in the app
X = df[['Internal Marks (Standardized)', 'Preboard Marks (Standardized)']]
y = df['Predicted Grade']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
st.success(" Model trained successfully!")

st.title("Student Grade Prediction App (KNN)")

st.header("Predict Student Grade")
internal_marks = st.number_input("Internal Marks (Standardized)", 0.0, 1.0, step=0.01)
preboard_marks = st.number_input("Preboard Marks (Standardized)", 0.0, 1.0, step=0.01)

if st.button("Predict Grade"):
    prediction = knn.predict([[internal_marks, preboard_marks]])
    st.success(f"Predicted Grade: {prediction[0]}")

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
