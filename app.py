import streamlit as st
import pickle
import pandas as pd

# List of teams and cities for the drop-down options
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title of the Streamlit app
st.title('IPL Win Predictor')

# Creating two columns for batting and bowling team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select the city for the match
selected_city = st.selectbox('Select host city', sorted(cities))

# Input field for target score
target = st.number_input('Target', min_value=0)

# Creating three columns for score, overs, and wickets input
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

# Predict button to calculate win probabilities
if st.button('Predict Probability'):
    # Calculate remaining runs, balls, and other necessary parameters
    runs_left = target - score
    balls_left = 120 - (overs * 6)  # 120 balls in total for a T20 match
    wickets_left = 10 - wickets
    crr = score / overs  # Current run rate
    rrr = (runs_left * 6) / balls_left  # Required run rate

    # Create the input data frame for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict the probabilities
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display the predictions
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
