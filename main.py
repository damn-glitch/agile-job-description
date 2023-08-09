import sys
import urllib
from urllib.parse import urlparse, parse_qs
import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate



def get_auth_url():
    url = f"https://www.linkedin.com/oauth/v2/authorization?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&state=123456&scope=r_liteprofile%20r_emailaddress"
    return url


# LinkedIn API and candidate search
CLIENT_ID = "77o34dxq91m1jr"
CLIENT_SECRET = "5ZfOsgI2T44a5fR9"
REDIRECT_URI = "https://jasaim.kz/auth/linkedin/callback.php"


def get_access_token(auth_code):
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    token_data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(token_url, data=token_data)
    response_data = response.json()
    access_token = response_data["access_token"]
    return access_token

def extract_keywords(job_description):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(job_description)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return keywords


def search_candidates(access_token, keywords):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    query = "software developer"  # Use a simple query to test the API's response
    url = f"https://api.linkedin.com/v2/search?keywords={query}&sort=relevance&count=1000"

    response = requests.get(url, headers=headers)
    response_data = response.json()
    candidates = response_data.get("elements", [])
    print(response_data)  # Add this line to print the API response data
    return candidates


def get_profile_url(member_id):
    return f"https://www.linkedin.com/in/{member_id}"

def rank_candidates(job_description, candidates):
    if not candidates:
        print("No candidates found.")
        return []

    candidate_profiles = [candidate.get("headline", "") for candidate in candidates]
    texts = [job_description] + candidate_profiles
    vectorizer = TfidfVectorizer()
    text_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    sorted_scores_indices = similarity_scores.argsort()[0][::-1]
    ranked_candidates = [(candidates[i], similarity_scores[0][i]) for i in sorted_scores_indices]
    return ranked_candidates


def main(auth_code):
    print("Auth code in main:", auth_code)  # Add this line
    access_token = get_access_token(auth_code)
    # Read job description from 'job.txt' file
    with open('job.txt', 'r') as file:
        job_description = file.read().strip()

    keywords = extract_keywords(job_description)
    candidates = search_candidates(access_token, keywords)
    ranked_candidates = rank_candidates(job_description, candidates)

    table_data = []
    for candidate, score in ranked_candidates:
        candidate_name = candidate.get("firstName", "") + " " + candidate.get("lastName", "")
        candidate_headline = candidate.get("headline", "")
        candidate_url = get_profile_url(candidate["id"])
        table_data.append([candidate_name, candidate_headline, candidate_url, f"{score * 100:.2f}%"])

    headers = ["Candidate name", "Headline", "Profile URL", "Percentage to suit a job"]
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))


if __name__ == "__main__":
    auth_url = get_auth_url()
    print("Please open the following URL in your browser and authorize the application:")
    print(auth_url)
    auth_code = input("Enter the authorization code: ")
    main(auth_code)

