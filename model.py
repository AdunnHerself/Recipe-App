import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

rf_class = data['model']
encoder = data['encoder']

def high_traffic_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=['calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings'])

    # Encode categorical feature 'category'
    input_df['category'] = encoder.fit_transform(input_df['category'])

    # Convert the DataFrame to a NumPy array for prediction
    input_data_reshaped = input_df.values

    # Print the input data to check
    print("Input data for prediction:", input_data_reshaped)

    # Make prediction using trained model
    prediction = rf_class.predict(input_data_reshaped)

    # Print the prediction to check
    print("Prediction result:", prediction)

    # Map numeric prediction to text
    if prediction[0] == 0:
        return 'This recipe will lead to low traffic'
    else:
        return 'This recipe will lead to high traffic'

def main():
    st.title('High Traffic Recipe Prediction Web App')

    calories = st.text_input("Amount of calories")
    carbohydrate = st.text_input("Amount of carbohydrate")
    sugar = st.text_input("Amount of sugar")
    protein = st.text_input("Amount of protein")
    category = st.selectbox("Select Category", ['Chicken', "Breakfast", "Beverages", "Lunch/Snacks", "Potato", "Pork", "Vegetable", "Dessert", "Meat", "One Dish Meal"])
    servings = st.selectbox('Select number of servings', [1, 2, 4, 6])

    # Code for prediction
    result = ''

    # Creating a button for prediction
    if st.button("Recipe Prediction Result"):
        try:
            # Convert inputs to the correct types
            input_data = [
                float(calories),
                float(carbohydrate),
                float(sugar),
                float(protein),
                category,
                int(servings)
            ]
            result = high_traffic_prediction(input_data)

        except ValueError as e:
            result = f"Error: {str(e)}"
            print(f"Error in prediction: {e}")

        st.success(result)

if __name__ == '__main__':
    main()