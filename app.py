import streamlit as st
import pandas as pd
import pickle


# Function to load model
def load_model():
    with open('saved_steps.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
rf_class = data['model']
encoder = data['encoder']

def high_traffic_prediction(input_data):
    input_df = pd.DataFrame([input_data], columns=['calories', 'carbohydrate', 'sugar', 'protein', 'category', 'servings'])
    input_df['category'] = encoder.fit_transform(input_df['category'])

    # Convert the DataFrame to a NumPy array for prediction
    input_data_reshaped = input_df.values

    # Make prediction using trained model
    prediction = rf_class.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'This recipe will lead to low traffic'
    else:
        return 'This recipe will lead to high traffic'

def main():
    st.set_page_config(page_title="High Traffic Recipe Prediction App", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "High Traffic Prediction", "Developer Info"])

    if page == "Home":
        st.title('Inspiration Behind the Project')
        st.write("""
        This project was inspired by my Datacamp certification examination. It's about a company Tasty Bytes, founded in 2020 amidst the Covid pandemic, their mission was clear: to provide inspiration when the world needed it most. 
        With a simple idea—helping people make the most of their limited supplies at home—they began as a search engine for recipes. 
        their goal was to offer creative solutions and ensure families could enjoy nutritious and varied meals despite the constraints.

        Their journey progressed, they discovered a crucial insight: selecting the right recipe can significantly impact traffic to their website. 
        They observed that showcasing a popular recipe could boost website traffic by up to 40%. 
        This realization highlighted a vital need—understanding which recipes would resonate most with our audience and drive engagement.

        This insight became the driving force behind the development of "High Traffic Recipe Prediction Web App." 
        They envisioned a tool that would empower restaurants and content creators to predict recipe popularity, harnessing data to make informed decisions. 
        Their aim is to enable businesses to anticipate customer preferences, enhance their menu offerings, and ultimately increase their success

        ### What the App Does
        The app allows users to input details about a recipe and predicts the traffic potential based on a pre-trained machine learning model. 
        It uses factors like nutritional content and category to make the prediction.

        ### How it Was Built
        - **Data Collection**: Datacamp provided the recipe dataset.
        - **Data Preprocessing**: Challenges that were faced include dealing with missing values, inconsistency in datatypes, outliers.
            dataprocessing techinques for filling missing values, correcting datatypes inconsistencies and dealing with outliers were applied
        - **Model Training**: Trained a Random Forest classifier to predict traffic potential,
            The model performance was evaluated on metrics like Accuracy, Precision and Recall. An accuarcy score of 76% was achieved,
                 A Precision score of 81% was achieved and a Recall score of 79% was achieved.

        - **App Development**: Built using Streamlit for the web interface and Pythons libraries.
        """)

    elif page == "High Traffic Prediction":
        st.title('High Traffic Recipe Prediction Web App')
        
        st.markdown(
            """
            <style>
            .big-font {
                font-size:50px !important;
                font-family: 'Arial', sans-serif;
                color: #4CAF50;
            }
            .medium-font {
                font-size:25px !important;
                font-family: 'Courier New', monospace;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<p class="big-font">Enter Recipe Details</p>', unsafe_allow_html=True)
        
        calories = st.text_input("Amount of calories")
        carbohydrate = st.text_input("Amount of carbohydrate")
        sugar = st.text_input("Amount of sugar")
        protein = st.text_input("Amount of protein")
        category = st.selectbox("Select Category", ['Chicken', "Breakfast", "Beverages", "Lunch/Snacks", "Potato", "Pork", "Vegetable", "Dessert", "Meat", "One Dish Meal"])
        servings = st.selectbox('Select number of servings', [1, 2, 4, 6])
        
        if st.button("Predict"):
            try:
                input_data = [
                    float(calories),
                    float(carbohydrate),
                    float(sugar),
                    float(protein),
                    category,
                    int(servings)
                ]
                result = high_traffic_prediction(input_data)

                # Display input details
                st.markdown('<p class="medium-font">Recipe Details</p>', unsafe_allow_html=True)
                st.write(f"**Calories:** {calories} kcal")
                st.write(f"**Carbohydrate:** {carbohydrate} g")
                st.write(f"**Sugar:** {sugar} g")
                st.write(f"**Protein:** {protein} g")
                st.write(f"**Category:** {category}")
                st.write(f"**Servings:** {servings}")
                #st.image(r"C:\Users\User\Desktop\Food Recipe\recipeImg.jpg", width=300)  # Placeholder for recipe image
                
                # Display prediction result last
                st.markdown('<p class="medium-font">Prediction Result</p>', unsafe_allow_html=True)
                st.success(result)
                
            except ValueError as e:
                st.error(f"Error: {str(e)}")

    elif page == "Developer Info":
        st.title('Developer Info')
        st.write("""
        Hi! I'm Lasisi Romoke, the developer of this app. I'm passionate about data science and building web applications 
        that provide valuable insights and functionality to users.

        If you'd like to get in touch with me, feel free to reach out via email or connect with me on social media.
        """)
        st.image("C:\\Users\\User\\Desktop\\Food Recipe\\AdunImg.jpeg", width=300)  
        st.write("Email: lasisiromoke4@gamil.com")
        st.write("[LinkedIn](https://www.linkedin.com/in/romoke-lasisi/)")
        #st.write("[GitHub](https://github.com/yourusername)")

if __name__ == '__main__':
    main()