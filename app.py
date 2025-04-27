import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    # Convert the teams and city to their respective indices
    batting_team_code = teams.index(batting_team)  # Get the index of the batting team
    bowling_team_code = teams.index(bowling_team)  # Get the index of the bowling team
    city_code = cities.index(selected_city)  # Get the index of the selected city

    # Calculate the remaining data
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Prepare the input dataframe with indices instead of names
    input_df = pd.DataFrame({
        'batting_team': [batting_team_code],  # Use the index of the team
        'bowling_team': [bowling_team_code],  # Use the index of the team
        'city': [city_code],  # Use the index of the city
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict the result
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Display the results
    st.header(f'{batting_team} - {round(win * 100)}%')
    st.header(f'{bowling_team} - {round(loss * 100)}%')
