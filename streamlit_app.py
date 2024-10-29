import streamlit as st
import pandas as pd
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


# read in file of job links
file = pd.read_csv('Internship_Links.csv')
job_links = file['Link']


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to scrape job description text from a URL
def get_job_description_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        st.warning(f"Could not retrieve job description from {url}: {e}")
        return ""
    
def get_applicant_type_from_description(job_description):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that identifies the intended applicant type for a job posting."},
        {"role": "user", "content": f"Based on the following job description, specify the intended applicant type as one of these options: 'High School', 'Undergraduate', 'Graduate', 'Professional'.\n\nJob Description: {job_description}\n\nReturn only the applicant type as a single word."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    applicant_type_text = response.choices[0].message['content'].strip()
    return applicant_type_text

def get_eligibility_type_from_description(job_description):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that identifies the citizenship eligibility for a job posting."},
        {"role": "user", "content": f"Based on the following job description, specify the intended citizenship type as one of these options: 'US Citizen Only', 'Permanent Resident', 'Visa Holder'.\n\nJob Description: {job_description}\n\nReturn only the citizenship type as a single word."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    citizenship_type_text = response.choices[0].message['content'].strip()
    return citizenship_type_text

# Function to get LLM-based similarity score with ChatCompletion
def get_similarity_score(resume_text, job_description):
    # Check if the job description's applicant type matches the user's selection
    job_applicant_type = get_applicant_type_from_description(job_description)
    if job_applicant_type.lower() != applicant_type.lower():
        return 0  # Return a low score if it doesn't match

    # If it matches, proceed with similarity scoring
    messages = [
        {"role": "system", "content": "You are a helpful assistant that evaluates resume and job description similarity."},
        {"role": "user", "content": f"Compare the following resume to the job description and rate the match on a scale of 1 to 10.\n- Resume: {resume_text}\n- Job Description: {job_description}\n\nReturn only the rating as an integer."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    score_text = response.choices[0].message['content'].strip()
    return int(score_text)

# Streamlit App
st.title("Resume and Job Description Matching")

# Upload the resume in PDF format
resume_pdf = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
citizenship = st.selectbox("Eligibility", 
    ["U.S. Citizen", "Permanent Resident", "Visa", "Other"])

applicant_type = st.selectbox("Applicant Type", [
    "High School", 
    "Undergraduate - Freshman", 
    "Undergraduate - Sophomore",
    "Undergraduate - Junior",
    "Undergraduate - Senior",
    "Graduate - Masters",
    "Graduate - PhD", 
    "Postdoctoral"
])

if resume_pdf:
    # Extract text from the resume PDF
    resume_text = extract_text_from_pdf(resume_pdf)
    st.write("**Resume Text:**")
    st.text(resume_text[:500] + "...")

    # Compare the resume with each job description
    st.write("### Job Matches")
    results = []
    for url in job_links:
        job_description_text = get_job_description_text(url)
        if job_description_text:
            similarity_score = get_similarity_score(resume_text, job_description_text)
            results.append((url, similarity_score))

    # Display the top three job matches
    top_matches = sorted(results, key=lambda x: x[1], reverse=True)[:3]
    for url, score in top_matches:
        st.write(f"**Job URL:** {url} | **Match Score:** {score}/10")

