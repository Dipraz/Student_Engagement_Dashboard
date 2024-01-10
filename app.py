import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import folium
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title="Student Engagement Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
@st.cache_data
def load_data():
    return pd.read_csv('mock_student_engagement_data.csv')

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
    st.title('Enrollment Trends Over Time - Dive Deep into Student Engagement')
    
    # Interactive time period selector with a distinct layout
    st.sidebar.header('Customize Your View')
    selected_years = st.sidebar.multiselect('Select Years', df['Year'].unique())
    selected_months = st.sidebar.multiselect('Select Months', df['Month'].unique())
    selected_weeks = st.sidebar.multiselect('Select Weeks', df['Week'].unique())

    filtered_data = df[df['Year'].isin(selected_years) & df['Month'].isin(selected_months) & df['Week'].isin(selected_weeks)]

    # Visually appealing line chart with interactive features
    fig = px.line(filtered_data, x='Year', y='Student_Count', title='Yearly Enrollment Trends')
    fig.update_layout(
        template='plotly_dark',  # Apply a dark theme for visual appeal
        xaxis_title='Year',
        yaxis_title='Student Count',
        hovermode='x unified'  # Enable unified hover interactions
    )
    st.plotly_chart(fig, use_container_width=True)  # Expand chart to full width

    # Additional interactive elements
    st.subheader('Explore Further with Interactive Insights')
    column1, column2 = st.columns(2)

    # Indentation corrected here:
    with column1:
        # Interactive data distribution analysis
        st.subheader('Explore Data Distributions')
        selected_column = st.selectbox('Select Column to Analyze', df.columns)
        st.write(df[selected_column].describe())
        st.bar_chart(df[selected_column])

        # Correlation matrix
        st.subheader('Correlation Matrix')
        corr_matrix = np.corrcoef(df.select_dtypes(include=[np.number]).dropna(axis=1).values.T)
        fig = px.imshow(corr_matrix, text_auto=True, aspect='equal', title='Correlation Matrix')
        st.plotly_chart(fig)

    with column2:
        # Key takeaways
        st.subheader('Key Insights from Enrollment Trends')
        st.write('- Enrollments have steadily increased over the past three years, with a notable surge in 2024.')
        st.write('- September consistently sees the highest enrollments, while December has the lowest.')
        st.write('- Week 2 consistently has higher enrollments compared to other weeks within a month.')

        # Potential factors and recommendations
        st.subheader('Factors and Recommendations')
        st.write('- Explore potential factors driving the observed trends, such as marketing campaigns, course offerings, or external events.')
        st.write('- Consider strategies to sustain enrollment growth, such as expanding online course options or offering flexible learning pathways.')
        st.write('- Investigate reasons for lower enrollments in December and potential initiatives to address them.')

if section == 'Geographical Distribution':
    st.title('Explore Global Reach: Where Students Engage')

    # Interactive map with hover effects and tooltips
    year_option = st.selectbox("Select Year", df['Year'].unique())  # Move year selection up for clarity
    map_type = st.radio("Select Map Type", ['Choropleth', 'Scatter Geo'])

    df_filtered = df[df['Year'] == year_option]  # Filter data based on selected year

    if map_type == 'Choropleth':
        map_fig = px.choropleth(df_filtered, locations='Country', locationmode='country names',
                                color='Student_Count', hover_name='Country',
                                hover_data=['Student_Count', 'Interest_Area'],
                                animation_frame='Year', title='Uncover Enrollment Trends Across the Globe')
    else:
        map_fig = px.scatter_geo(df_filtered, locations='Country', locationmode='country names',
                                size='Student_Count', color='Interest_Area', hover_name='Country',
                                title='Enrollment Distribution by Country')

    map_fig.update_layout(coloraxis_colorbar=dict(title='Student Count'),
                          margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(map_fig, use_container_width=True)

    # Key takeaways and insights
    st.subheader('Key Insights from Global Distribution')
    st.write('- Top 3 countries with the highest enrollments:')
    st.write(df_filtered.groupby('Country')['Student_Count'].sum().nlargest(3))
    st.write('- Countries with the most diverse interest areas:')
    st.write(df_filtered.groupby('Country')['Interest_Area'].nunique().nlargest(3))
    st.write('- Year-over-year growth in specific regions:')
    # ... (add insights based on your data)

    # Interactive comparison with previous year
    if year_option != df['Year'].min():
        prev_year_data = df[df['Year'] == (year_option - 1)]
        comparison_fig = go.Figure(data=[
            go.Bar(name=str(year_option), x=df_filtered['Country'], y=df_filtered['Student_Count']),
            go.Bar(name=str(year_option - 1), x=prev_year_data['Country'], y=prev_year_data['Student_Count'])
        ])
        comparison_fig.update_layout(barmode='group', title='Enrollment Comparison with Previous Year')
        st.plotly_chart(comparison_fig)

if section == 'Course Breakdown':
    st.title('Uncover Course Popularity: What Students Choose')

    # Interactive filtering and sorting (using available columns)
    year_option = st.selectbox("Select Year", df['Year'].unique())
    degree_option = st.selectbox("Filter by Degree Type (optional)", df['Degree_Type'].unique(), index=0)
    if degree_option != 'All':
        df_filtered = df[df['Degree_Type'] == degree_option]
    else:
        df_filtered = df

    sort_option = st.radio("Sort Courses By", ['Enrollment Count (Descending)', 'Alphabetically'])
    if sort_option == 'Enrollment Count (Descending)':
        df_filtered = df_filtered.sort_values(by='Student_Count', ascending=False)
    else:
        df_filtered = df_filtered.sort_values(by='Course')

    # Dynamic chart creation with eye-catching colors and hover effects
    view_option = st.radio("Visualize Course Breakdown", ['Bar Chart', 'Pie Chart'])
    if view_option == 'Bar Chart':
        course_fig = px.bar(df_filtered, x='Course', y='Student_Count',
                            color='Degree_Type', color_discrete_sequence=px.colors.qualitative.Pastel,
                            hover_name='Course', hover_data=['Student_Count', 'Year'],
                            title='Enrollments by Course')
    else:
        course_fig = px.pie(df_filtered, values='Student_Count', names='Course',
                            color='Degree_Type', color_discrete_sequence=px.colors.qualitative.Set2,
                            hover_name='Course', hover_data=['Student_Count', 'Year'],
                            title='Enrollment Distribution by Course')

    course_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), title_font_size=20)
    st.plotly_chart(course_fig, use_container_width=True)

    # Key insights and recommendations (adjusted for available data)
    st.subheader('Key Insights from Course Data')
    st.write('- Top 3 courses with the highest enrollments:')
    st.write(df.groupby('Course')['Student_Count'].sum().nlargest(3))
    st.write('- Degree types with the most popular courses:')
    st.write(df.groupby('Degree_Type')['Student_Count'].sum().nlargest(3))
    # ... (add more insights based on your data)

    # Recommendations for course prioritization or promotion
    st.subheader('Recommendations:')
    st.write('- Consider promoting courses with high enrollment potential:')
    # ... (provide recommendations based on your analysis)

if section == 'Student Demographics':
    st.title("Uncover Your Student Body: Who's Learning?")  # More engaging title

    # Interactive filtering and year comparison
    year_option = st.selectbox("Select Year", df['Year'].unique())
    df_filtered = df[df['Year'] == year_option]

    # Dynamic chart creation with eye-catching colors and hover effects
    gender_fig = px.pie(df_filtered, names='Gender', values='Student_Count',
                        color_discrete_sequence=px.colors.sequential.RdBu,
                        hover_name='Gender', hover_data=['Student_Count'],
                        title='Gender Distribution')
    degree_fig = px.bar(df_filtered, x='Degree_Type', y='Student_Count',
                        color='Degree_Type', color_discrete_sequence=px.colors.qualitative.Pastel,
                        hover_name='Degree_Type', hover_data=['Student_Count'],
                        title='Enrollments by Degree Type')

    # Arrange charts for visual clarity
    col1, col2 = st.columns(2)
    col1.plotly_chart(gender_fig, use_container_width=True)
    col2.plotly_chart(degree_fig, use_container_width=True)

    # Key insights and recommendations
    st.subheader('Key Insights from Demographics')
    st.write('- Gender breakdown for the selected year:')
    st.write(df_filtered['Gender'].value_counts().to_frame().style.format({'Gender': '{:.2%}'}))
    st.write('- Top 3 degree types by enrollment:')
    st.write(df_filtered.groupby('Degree_Type')['Student_Count'].sum().nlargest(3))
    # ... (add more insights based on your data)

    # Recommendations for targeted outreach or resource allocation
    st.subheader('Recommendations:')
    st.write('- Consider strategies to attract underrepresented gender groups:')
    # ... (provide recommendations based on your analysis)

if section == 'Interest Areas':
    st.title('Explore Evolving Interests: What Your Students Crave')  # Engaging title

    # Interactive filtering with multiple options
    year_slider = st.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()),
                            (int(df['Year'].min()), int(df['Year'].max())))
    interest_area_option = st.multiselect("Filter by Interest Area", df['Interest_Area'].unique())
    df_filtered = df[(df['Year'] >= year_slider[0]) & (df['Year'] <= year_slider[1])]
    if interest_area_option:
        df_filtered = df_filtered[df_filtered['Interest_Area'].isin(interest_area_option)]

    # Dynamic Chart with Eye-Catching Visuals
    interest_fig = px.bar(df_filtered, x='Year', y='Student_Count', color='Interest_Area',
                          color_discrete_sequence=px.colors.qualitative.Bold,  # Bold color palette
                          hover_name='Interest_Area', hover_data=['Year', 'Student_Count'],
                          title='Interest Area Trends Over Time')
    interest_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))  # Adjust margins

    # Interactive Sunburst Chart Highlighting Interest Area Distribution
    sunburst_fig = px.sunburst(df_filtered, path=['Year', 'Interest_Area'], values='Student_Count',
                               color='Interest_Area', color_discrete_sequence=px.colors.qualitative.Pastel)
    sunburst_fig.update_layout(margin=dict(t=40, b=20))

    # Interactive Line Chart Showcasing Trends with Smoothing
    line_fig = px.line(df_filtered, x='Year', y='Student_Count', color='Interest_Area', line_group='Interest_Area',
                       hover_name='Interest_Area', hover_data=['Year', 'Student_Count'],
                       title='Interest Area Trends with Smoothing')
    line_fig.update_traces(mode='lines+markers')  # Add markers for clarity
    line_fig.update_layout(margin=dict(t=40, b=20))

    # Display the charts side-by-side for visual comparison
    st.plotly_chart(interest_fig, use_container_width=True)
    st.plotly_chart(sunburst_fig, use_container_width=True)
    st.plotly_chart(line_fig, use_container_width=True)

    # Key Insights and Recommendations (tailored to the visuals)
    st.subheader('Key Insights from Interest Data')
    st.write('- Explore the dynamic bar chart to visualize trends over time and hover for details.')
    st.write('- Uncover hierarchical relationships and distribution patterns with the interactive sunburst chart.')
    st.write('- Observe trends with smoothing using the line chart.')

if section == 'Additional Insights':
    st.title('Discover Further Insights')

    insight_type = st.sidebar.selectbox('Choose Insight Type', ['Correlations', 'Key Drivers', 'External Factors'])
    visualization_type = st.sidebar.selectbox('Choose Visualization', ['Bar Chart', 'Scatter Plot', 'Map'])

    st.write("**Uncover hidden patterns and meaningful relationships within the student data.**")
    st.write("Leverage interactive visualizations and insights to make informed decisions and drive positive student outcomes.")

    if insight_type == 'Correlations':
        numeric_cols = df.select_dtypes(include=['int32', 'float64']).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        st.write("Correlation Matrix:")
        st.dataframe(corr_matrix)

    elif insight_type == 'Key Drivers':
        X = df.drop('Student_Count', axis=1)
        y = df['Student_Count']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(handle_unknown='ignore')
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_train = pd.concat([X_train.drop(categorical_cols, axis=1), 
                            pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(), index=X_train.index)], axis=1)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

        feature_importances = model.feature_importances_
        sorted_idx = np.argsort(feature_importances)[::-1]
        
        # Display key drivers
        st.subheader("Key Drivers of Student Enrollment")
        st.write("The most influential features in predicting student enrollment are:")
        for i in range(5):  # Display top 5 features, adjust as needed
            st.write(f"{i+1}. {X_train.columns[sorted_idx[i]]} ({feature_importances[sorted_idx[i]]:.4f})")

    elif insight_type == 'External Factors':
        if {'Latitude', 'Longitude', 'Country', 'Student_Count'}.issubset(df.columns):
            map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=3)

            for index, row in df.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=row['Student_Count'] * 0.01,
                    popup=f"Country: {row['Country']}<br>Student Count: {row['Student_Count']}",
                    color='blue',
                    fill=True,
                    fill_color='lightblue'
                ).add_to(map)
            st.write("Map showing Student Counts based on External Factors:")
            st_folium(map)
        else:
            st.error("Required columns for External Factors analysis are missing.")
            
if section == 'Monitoring and Alerts':
    st.title('Stay Ahead of the Curve: Real-Time Monitoring and Alerts')  # Engaging title

    # Interactive Alert Threshold Sliders
    min_student_count_threshold = st.slider("Minimum Student Count Threshold", 0, df['Student_Count'].max(), 100)
    max_gender_imbalance_threshold = st.slider("Maximum Gender Imbalance Threshold (% difference)", 0, 100, 5)

    # Calculate metrics and trigger alerts based on thresholds
    current_student_count = df['Student_Count'].sum()
    gender_counts = df['Gender'].value_counts()
    gender_imbalance_percent = (abs(gender_counts.max() - gender_counts.min()) / df.shape[0]) * 100

    if current_student_count < min_student_count_threshold:
        st.error("Alert: Student count has fallen below the threshold!")
        st.write("Current student count:", current_student_count)

    if gender_imbalance_percent > max_gender_imbalance_threshold:
        st.warning("Alert: Gender imbalance is exceeding the threshold!")
        st.write("Current gender imbalance:", gender_imbalance_percent, "%")

    # Visualize key metrics with interactive charts
    st.subheader("Key Metrics at a Glance")
    st.metric("Total Student Count", current_student_count)
    gender_chart = px.pie(df, values='Student_Count', names='Gender', title='Gender Distribution')
    st.plotly_chart(gender_chart)

    # Linear Regression Model for Future Predictions
    df['Year'] = df['Year'].astype(int)  # Convert Year to integer
    X = df[['Year']]
    y = df['Student_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    latest_year = df['Year'].max()
    future_years = range(latest_year + 1, latest_year + 4)
    future_df = pd.DataFrame({'Year': future_years})
    future_predictions = model.predict(future_df)
    st.subheader("Future Predictions of Student Count")
    future_prediction_df = pd.DataFrame({'Year': future_years, 'Predicted Student Count': future_predictions.astype(int)})
    st.dataframe(future_prediction_df)

    # Simple visualization of predicted student counts over future years
    fig = px.line(future_prediction_df, x='Year', y='Predicted Student Count', title='Predicted Student Count over Future Years')
    st.plotly_chart(fig)
