import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import langchain.globals as lcg

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'platform' not in st.session_state:
    st.session_state.platform = 'education'

# Set verbose to True or False based on your requirements
lcg.set_verbose(True)

# Set up the model and prompt template for AI insights
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDA27FYUw_BM6A6GNqOeBQRkScY9_6fv2E'  # Replace with your secure API key
generation_config = {"temperature": 0.6, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

# Prompt templates for each platform
prompt_template = {
    'education': PromptTemplate(
        input_variables=['domain', 'place'],
        template="Education Insights:\n"
                 "Domain of Interest: {domain}\n"
                 "Place: {place}\n"
                 "Provide a list of colleges, schools, or institutions nearby {place} that specialize in the domain of {domain}. Include brief insights into why they might be suitable."
    ),
    'health': PromptTemplate(
        input_variables=['symptoms', 'place'],
        template="Health Insights:\n"
                 "Symptoms: {symptoms}\n"
                 "Location: {place}\n"
                 "Based on the symptoms provided, recommend suitable English medicines, Ayurvedic medicines, and precautions to take. Also, provide a list of hospitals or clinics nearby {place} where the user can seek medical advice."
    ),
    'agriculture': PromptTemplate(
        input_variables=['agriculture_data'],
        template="Agriculture Insights:\n"
                 "Provide an analysis report based on the following data:\n"
                 "Agriculture Data: {agriculture_data}\n"
                 "The report should include actionable insights for improving agricultural practices."
    ),
    'domain_brief': PromptTemplate(
        input_variables=['domain'],
        template="Provide a brief overview of the domain: {domain}. Describe its importance, applications, and current trends."
    )
}

# Create a Runnable chain for AI insights for each platform
chain = {
    platform: RunnableLambda(lambda inputs, platform=platform: prompt_template[platform].format(**inputs)) | model
    for platform in prompt_template.keys()
}

# Authentication functions
def login(username, password):
    if username in st.session_state.users and st.session_state.users[username] == password:
        st.session_state.logged_in = True
        st.session_state.page = 'dashboard'
        st.experimental_set_query_params(page=st.session_state.page)
    else:
        st.error("Invalid username or password")

def signup(username, password):
    if username in st.session_state.users:
        st.error("Username already exists")
    else:
        st.session_state.users[username] = password
        st.session_state.logged_in = True
        st.session_state.page = 'dashboard'
        st.experimental_set_query_params(page=st.session_state.page)

def logout():
    st.session_state.logged_in = False
    st.session_state.page = 'login'
    st.experimental_set_query_params(page=st.session_state.page)

# Login page
if st.session_state.page == 'login':
    st.title("E-Village Platform")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        login(username, password)
    if st.button("Signup"):
        signup(username, password)

# Dashboard page
elif st.session_state.page == 'dashboard':
    st.title("E-Village Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Platforms")
    platform_choice = st.sidebar.radio("Choose a platform:", ('education', 'health', 'agriculture'))
    st.session_state.platform = platform_choice

    if st.session_state.platform == 'education':
        st.header("Education Platform")
        with st.form("education_form"):
            domain = st.text_input("Domain of Interest", placeholder="E.g., Data Science, Engineering, Arts")
            place = st.text_input("Place", placeholder="E.g., Bangalore, Delhi")
            submit_button = st.form_submit_button("Get Education Insights")
        
        if submit_button:
            # First, get a brief idea about the domain
            domain_input = {'domain': domain}
            domain_brief = chain['domain_brief'].invoke(domain_input)
            
            # Display the brief idea about the domain
            st.subheader("Brief Overview of the Domain")
            st.markdown(domain_brief)
            
            # Then, get education insights based on the domain and place
            input_data = {
                'domain': domain,
                'place': place
            }
            
            # Fetch education insights using the AI model
            education_insights = chain['education'].invoke(input_data)
            
            # Display the AI-generated recommendations
            st.subheader("Recommended Institutions")
            st.markdown(education_insights)
            
    elif st.session_state.platform == 'health':
        st.header("Health Platform")
        with st.form("health_form"):
            symptoms = st.text_area("Enter your symptoms", placeholder="E.g., headache, fever, sore throat")
            place = st.text_input("Location", placeholder="E.g., Bangalore, Mumbai")
            submit_button = st.form_submit_button("Get Health Insights")
        
        if submit_button:
            input_data = {
                'symptoms': symptoms,
                'place': place
            }
            
            # Fetch health insights using the AI model
            health_insights = chain['health'].invoke(input_data)
            
            # Display the AI-generated recommendations
            st.subheader("Health Recommendations")
            st.markdown(health_insights)
            
    elif st.session_state.platform == 'agriculture':
        st.header("Agriculture Platform")
        with st.form("agriculture_form"):
            agriculture_data = st.text_area("Agriculture Data", placeholder="E.g., crop yield, types of crops, etc.")
            submit_button = st.form_submit_button("Get Agriculture Insights")
        
        if submit_button:
            input_data = {'agriculture_data': agriculture_data}
            agriculture_insights = chain['agriculture'].invoke(input_data)
            st.subheader("Agriculture Insights")
            st.markdown(agriculture_insights)

    if st.button("Logout"):
        logout()
