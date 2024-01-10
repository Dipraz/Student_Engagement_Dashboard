import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Load data (assuming df is the DataFrame you have prepared)
@st.cache_data
def load_data():
    return pd.read_csv(r'C:\Users\dip07\mock_student_engagement_data.csv')

df = load_data()

st.sidebar.title('Navigation')
section = st.sidebar.radio("Go to", ['Home', 'Enrollment Trends', 'Geographical Distribution', 'Course Breakdown', 'Student Demographics', 'Interest Areas', 'Additional Insights', 'Monitoring and Alerts'])

if section == 'Home':
    st.title('Student Engagement Dashboard')
    st.write('A comprehensive view of student enrollment trends, demographics, and more.')
if section == 'Enrollment Trends':
    st.title('Enrollment Trends Over Time')
    
    # Time period selector
    years = df['Year'].unique()
    selected_years = st.multiselect('Select Year', years, default=years)
    filtered_data = df[df['Year'].isin(selected_years)]

    # Line chart
    fig = px.line(filtered_data, x='Year', y='Student_Count', title='Yearly Enrollment Trends')
    st.plotly_chart(fig)
if section == 'Geographical Distribution':
    st.title('Geographical Distribution of Enrollments')
    
    # World Map Heatmap
    fig = px.choropleth(df, locations='Country', locationmode='country names', color='Student_Count')
    st.plotly_chart(fig)
if section == 'Course Breakdown':
    st.title('Course Breakdown')
    
    # Course distribution chart
    fig = px.bar(df, x='Course', y='Student_Count', title='Enrollment by Course')
    st.plotly_chart(fig)
if section == 'Student Demographics':
    st.title('Student Demographics')
    
    # Gender Distribution
    gender_fig = px.pie(df, names='Gender', title='Gender Distribution')
    st.plotly_chart(gender_fig)

    # Degree Type Distribution
    degree_fig = px.bar(df, x='Degree_Type', y='Student_Count', title='Enrollment by Degree Type')
    st.plotly_chart(degree_fig)
if section == 'Interest Areas':
    st.title('Interest Areas')
    
    # Stacked Bar Chart for Interest Areas
    interest_fig = px.bar(df, x='Year', y='Student_Count', color='Interest_Area', title='Interest Area Trends')
    st.plotly_chart(interest_fig)
if section == 'Additional Insights':
    st.title('Additional Insights')
    # Implement insights like retention rates, demographic trends
if section == 'Monitoring and Alerts':
    st.title('Monitoring and Alerts')
    # Setup for monitoring and real-time alerts
