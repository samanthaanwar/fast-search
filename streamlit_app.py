import streamlit as st
import pandas as pd
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# # Load the job descriptions dataset
# @st.cache_data
# def load_job_descriptions():
#     # Update with the path to your dataset
#     return pd.read_csv("Internship_Links.csv")

# # Function to get LLM-based similarity score
# def get_similarity_score(resume_text, job_description):
#     prompt = f"""
#     Compare the following resume to the job description and rate the match on a scale of 1 to 10.
#     - Resume: {resume_text}
#     - Job Description: {job_description}
    
#     Return only the rating as an integer.
#     """
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=10
#     )
#     return int(response.choices[0].text.strip())

# # Streamlit App
# st.title("Resume and Job Description Matching")

# # Allow user to upload resume
# uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])

# # Load job descriptions data
# job_descriptions = load_job_descriptions()

# if uploaded_file:
#     # Read the uploaded resume file
#     resume_text = uploaded_file.read().decode("utf-8")
#     st.write("**Resume Text:**")
#     st.text(resume_text)

#     # Compare resume with each job description in the dataset
#     st.write("### Job Matches")
#     results = []
#     for _, row in job_descriptions.iterrows():
#         job_title = row["job_title"]
#         job_description = row["job_description"]
#         similarity_score = get_similarity_score(resume_text, job_description)
#         results.append((job_title, similarity_score))
    
#     # Display the top job matches
#     top_matches = sorted(results, key=lambda x: x[1], reverse=True)[:5]
#     for title, score in top_matches:
#         st.write(f"**{title}**: Match Score = {score}/10")


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

# Function to get LLM-based similarity score with ChatCompletion
def get_similarity_score(resume_text, job_description):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that evaluates resume and job description similarity."},
        {"role": "user", "content": f"Compare the following resume to the job description and rate the match on a scale of 1 to 10.\n- Resume: {resume_text}\n- Job Description: {job_description}\n\nReturn only the rating as an integer."}
    ]
    response = openai.ChatCompletion.acreate(
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

