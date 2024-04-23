import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Loading the saved model
with open("lgbm_model4.pkl", "rb") as file:
    loaded_model = pickle.load(file)

def cwd_prediction(input_data):
    # Convert input data to numpy array and ensure all inputs are floats
    input_data_as_floats = np.array(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_floats.reshape(1, -1)

    # Making the prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        st.image('deern.png')
        return 'County CWD is Negative'
    else:
        st.image('deerp.png')
        return 'County CWD is Positive'

def main():
    st.title('Welcome to the CWD County-based Prediction Web App')
    image = Image.open("CWD1.png")
    st.image(image, use_column_width=True)
    st.header("User Input:")

    # Collect input features
    Cervid = st.text_input('Number of Cervid Facilities(0-9)')
    Harvest = st.text_input('Total Harvest(5-9950)')
    Hunting = st.selectbox('Hunting Enclosures', ['0', '0.5', '1'])
    Baiting = st.selectbox('Baiting', ['0', '0.5', '1'])
    Feeding = st.selectbox('Feeding', ['0', '0.5', '1'])
    Carcass = st.selectbox('Whole Carcass Import', ['0', '0.5'])
    Urine = st.selectbox('Urine Lures', ['0.0', '0.5', '1.0'])
    Captive = st.selectbox('Captive status', ['0', '1'])
    Forest = st.text_input('Forest (0.007-0.89)')
    Clay = st.text_input('Clay (Average Percent(5-31)')
    Streams = st.text_input('Streams (Average distance to the nearest water(1125-41430)')

    # Convert input data to numeric when button is pressed
    if st.button('CWD Test Result'):
        input_data = [Cervid, Harvest, Hunting, Baiting, Feeding, Carcass, Urine, Captive, Forest, Clay, Streams]
        diagnosis = cwd_prediction(input_data)
        st.success(diagnosis)

if __name__ == '__main__':
    main()
