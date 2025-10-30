import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the data
df = pd.read_csv('EDA_Formatted_Data.csv')

# Features and target
X = df[['Internal Marks (Standardized)', 'Preboard Marks (Standardized)']]
y = df['Predicted Grade']

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # Default k=5
knn.fit(X, y)

# Save the model using pickle
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Model trained and saved successfully.")
