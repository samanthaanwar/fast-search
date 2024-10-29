import streamlit as st
import pandas as pd
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Load the job descriptions dataset
@st.cache_data
def load_job_descriptions():
    # Update with the path to your dataset
    return pd.read_csv("Internship_Links.csv")

# Function to get LLM-based similarity score
def get_similarity_score(resume_text, job_description):
    prompt = f"""
    Compare the following resume to the job description and rate the match on a scale of 1 to 10.
    - Resume: {resume_text}
    - Job Description: {job_description}
    
    Return only the rating as an integer.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=10
    )
    return int(response.choices[0].text.strip())

# Streamlit App
st.title("Resume and Job Description Matching")

# Allow user to upload resume
uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])

# Load job descriptions data
job_descriptions = load_job_descriptions()

if uploaded_file:
    # Read the uploaded resume file
    resume_text = uploaded_file.read().decode("utf-8")
    st.write("**Resume Text:**")
    st.text(resume_text)

    # Compare resume with each job description in the dataset
    st.write("### Job Matches")
    results = []
    for _, row in job_descriptions.iterrows():
        job_title = row["job_title"]
        job_description = row["job_description"]
        similarity_score = get_similarity_score(resume_text, job_description)
        results.append((job_title, similarity_score))
    
    # Display the top job matches
    top_matches = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    for title, score in top_matches:
        st.write(f"**{title}**: Match Score = {score}/10")

