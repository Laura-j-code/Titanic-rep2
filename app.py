import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Titanic Survival prediction Web App')

st.write("Enter the feature values below:")

# Example: if you have 4 features (adjust according to your dataset)
feature1 = st.number_input('Passennger ID')
feature2 = st.number_input('Pclass')
feature3 = st.number_input('Sex')
feature4 = st.number_input('Age')
feature5 = st.number_input('Embarked')
feature6 = st.number_input('Family_size')

# Collect inputs into array
input_data = np.array([[feature1, feature2, feature3, feature4,feature5,feature6]])

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Class: {prediction[0]}')
