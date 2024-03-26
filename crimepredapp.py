from base64 import b64encode
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
# Setting background image for web application
def get_base64_of_binfile(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return b64encode(data).decode()
 
 
def set_bg_page(png_file):
    bin_str = get_base64_of_binfile(png_file)
    page_bg_img = '''
    <style>
    .st-emotion-cache-l9bjmx p{
    color:red;
    }
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
   
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
 
set_bg_page("citi1.png")
label_encoder = LabelEncoder()

# Load the label encoder
labels = pd.read_csv("label.csv")
y = labels['Primary Type']
label_encoder.fit(y)

# Load the trained model
with open('crimeprediction2.pkl', 'rb') as f:
    model = pickle.load(f)

# Dropdown options
months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
seasons = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
hours = {i: f'{i}:00 - {i+1}:00' for i in range(24)}
hour_categories = {0: 'Late Night (12am-6am)', 1: 'Morning (6am-12pm)', 2: 'Afternoon (12pm-6pm)', 3: 'Evening (6pm-12am)'}
arrest = {0: 'No', 1: 'Yes'}
domestic = {0: 'No', 1: 'Yes'}
beats = {1: 'Beat 1', 2: 'Beat 2', 3: 'Beat 3'}  
years = {2012: 2012, 2013: 2013, 2014: 2014, 2015: 2015, 2016: 2016, 2017: 2017, 2018: 2018, 2019: 2019, 2020: 2020, 2021: 2021}

# Dropdown widgets for each feature
month = st.selectbox('Month', options=list(months.values()))
season = st.selectbox('Season', options=list(seasons.values()))
hour = st.selectbox('Hour', options=list(hours.values()))
hour_category = st.selectbox('Hour Category', options=list(hour_categories.values()))
arrest_val = st.selectbox('Arrest', options=list(arrest.values()))
domestic_val = st.selectbox('Domestic', options=list(domestic.values()))
beat = st.selectbox('Beat', options=list(beats.values()))
year = st.selectbox('Year', options=list(years.values()))

# Create a button to trigger the prediction
if st.button("Predict"):
    # Encode the selected values
    month_encoded = list(months.keys())[list(months.values()).index(month)]
    season_encoded = list(seasons.keys())[list(seasons.values()).index(season)]
    hour_encoded = list(hours.keys())[list(hours.values()).index(hour)]
    hour_category_encoded = list(hour_categories.keys())[list(hour_categories.values()).index(hour_category)]
    arrest_encoded = list(arrest.keys())[list(arrest.values()).index(arrest_val)]
    domestic_encoded = list(domestic.keys())[list(domestic.values()).index(domestic_val)]
    beat_encoded = list(beats.keys())[list(beats.values()).index(beat)]
    year_encoded = list(years.keys())[list(years.values()).index(year)]
    
    # Create a DataFrame with the selected values
    input_data = pd.DataFrame({
        'month': [month_encoded],
        'season': [season_encoded],
        'hour': [hour_encoded],
        'hour_categorized': [hour_category_encoded],
        'Arrest': [arrest_encoded],
        'Domestic': [domestic_encoded],
        'Beat': [beat_encoded],
        'Year': [year_encoded]
    })
    
    # Predict the primary type
    input_data_encoded = input_data.values.reshape(1, -1)
    y_pred_encoded = model.predict(input_data_encoded)
    prediction = label_encoder.inverse_transform(y_pred_encoded)[0]
    # st.write(f"The predicted primary type of crime is: {prediction}")
    st.markdown(f"<h4 style='text-align: center; color: #941731;'>The predicted primary type of crime is:{prediction}</h4>", unsafe_allow_html=True)
