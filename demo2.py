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

# Initialize selected_years, selected_months, and selected_weeks with default values
selected_years = df['Year'].unique()
selected_months = df['Month'].unique()
selected_weeks = df['Week'].unique()

if section == 'Home':
    st.title('Student Engagement Dashboard')
    st.write('A comprehensive view of student enrollment trends, demographics, and more.')

if section == 'Enrollment Trends':
    st.title('Enrollment Trends Over Time')
    
    # Time period selector
    selected_years = st.multiselect('Select Year', selected_years, default=selected_years)
    selected_months = st.multiselect('Select Month', selected_months, default=selected_months)
    selected_weeks = st.multiselect('Select Week', selected_weeks, default=selected_weeks)
    
    filtered_data = df[df['Year'].isin(selected_years) & df['Month'].isin(selected_months) & df['Week'].isin(selected_weeks)]

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
    
    # Retention Rates
    st.subheader('Retention Rates')
    
    # Calculate retention rates or load relevant data
    # For example, you can calculate retention rates for different cohorts
    
    # Sample retention rate calculation for illustration
    def calculate_retention_rate(df, selected_years, selected_months, selected_weeks):
        # Filter data for the selected period
        filtered_data = df[df['Year'].isin(selected_years) & df['Month'].isin(selected_months) & df['Week'].isin(selected_weeks)]
        
        # Calculate the number of students at the beginning and end of the period
        start_period_students = df[df['Year'] == min(selected_years)]['Student_Count'].sum()
        end_period_students = filtered_data['Student_Count'].sum()
        
        # Calculate retention rate
        retention_rate = (end_period_students / start_period_students) * 100
        
        return round(retention_rate, 2)
    
    retention_rate = calculate_retention_rate(df, selected_years, selected_months, selected_weeks)
    
    # Display the insights using st.write() or st.plotly_chart() as needed
    st.write("Retention rate for the selected period:", retention_rate)


    # Demographic Trends
    st.subheader('Demographic Trends')
    
    # Display demographic insights such as gender distribution, degree type distribution, etc.
    # You can create charts or tables to display demographic trends
    
    # Gender Distribution
    gender_fig = px.pie(df, names='Gender', title='Gender Distribution')
    st.plotly_chart(gender_fig)

if section == 'Monitoring and Alerts':
    st.title('Monitoring and Alerts')
    
    # Setup for monitoring and real-time alerts
    st.subheader('Alert Configuration')
    
    # Include input fields or configuration options for setting up alerts
    alert_threshold = st.number_input("Alert Threshold", min_value=0, step=1, value=10)
    
    # Simulated data for demonstration
    current_value = 15  # Replace with actual data source
    
    # Display alerts or status information based on user settings
    if current_value > alert_threshold:
        st.warning("Alert: Current value exceeds the threshold!")
    else:
        st.success("No alerts at the moment. Everything is within the threshold.")
