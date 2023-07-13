import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# Step 1: Create a local database of random people
# This list of dictionaries represents a sample database of candidate profiles.
local_database = [
    {
        "id": 1,
        "name": "Alice Smith",
        "profile_url": "local:/1",
        "description": "Software engineer with experience in web development, Python, and JavaScript.",
    },
    {
        "id": 2,
        "name": "Bob Johnson",
        "profile_url": "local:/2",
        "description": "Digital marketer specializing in social media management and content creation.",
    },
    {
        "id": 3,
        "name": "Charlie Williams",
        "profile_url": "local:/3",
        "description": "Data scientist skilled in machine learning, data analysis, and visualization using Python and R.",
    },
    {
        "id": 4,
        "name": "Diana Brown",
        "profile_url": "local:/4",
        "description": "Product manager with a background in software development and experience leading cross-functional teams.",
    },
    {
        "id": 5,
        "name": "Eva Jones",
        "profile_url": "local:/5",
        "description": "Graphic designer proficient in Adobe Creative Suite, focusing on branding, print design, and digital media.",
    },
    {
        "id": 6,
        "name": "Frank Miller",
        "profile_url": "local:/6",
        "description": "UX designer specializing in user research, wireframing, and prototyping for web and mobile applications.",
    },
    {
        "id": 7,
        "name": "Grace Wilson",
        "profile_url": "local:/7",
        "description": "DevOps engineer experienced in CI/CD, cloud infrastructure, and containerization using Docker and Kubernetes.",
    },
    {
        "id": 8,
        "name": "Hannah Moore",
        "profile_url": "local:/8",
        "description": "Technical writer skilled in creating documentation, user guides, and API references for software products.",
    },
    {
        "id": 9,
        "name": "Ivan Taylor",
        "profile_url": "local:/9",
        "description": "Front-end developer with a strong focus on responsive design, HTML, CSS, and JavaScript frameworks like React and Angular.",
    },
    {
        "id": 10,
        "name": "Jack Anderson",
        "profile_url": "local:/10",
        "description": "Back-end developer experienced in designing and implementing RESTful APIs, database design, and server-side programming using Node.js, Django, and Ruby on Rails.",
    },
    {
        "id": 11,
        "name": "Kendra Thomas",
        "profile_url": "local:/11",
        "description": "Full-stack developer proficient in both front-end and back-end technologies, with a passion for creating scalable and performant web applications.",
    },
    {
        "id": 12,
        "name": "Liam Jackson",
        "profile_url": "local:/12",
        "description": "Mobile app developer skilled in developing native and cross-platform applications for iOS and Android using Swift, Kotlin, and React Native.",
    },
]


# Step 2: Process job description
# This function takes a job description text as input, tokenizes it, and extracts keywords using the spaCy library.
def extract_keywords(job_description):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(job_description)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return keywords


# Step 3: Search local database for candidates
# This function takes a list of keywords and returns the entire local_database as the search result (no filtering is performed).
def search_candidates(keywords):
    return local_database


# Step 4: Score and rank candidates
# This function takes a job description and a list of candidates, calculates the cosine similarity between the job description and each candidate's description, and returns a list of ranked candidates.
def rank_candidates(job_description, candidates):
    candidate_profiles = [candidate['description'] for candidate in candidates]
    texts = [job_description] + candidate_profiles
    vectorizer = TfidfVectorizer()
    text_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    sorted_scores_indices = similarity_scores.argsort()[0][::-1]
    ranked_candidates = [(candidates[i], similarity_scores[0][i]) for i in sorted_scores_indices]
    return ranked_candidates

# Main function to read job description from a file, extract keywords, search for candidates, rank them, and print the results in a tabular format.
def main():
    # Read job description from 'job.txt' file
    with open('job.txt', 'r') as file:
        job_description = file.read().strip()

    keywords = extract_keywords(job_description)
    candidates = search_candidates(keywords)
    ranked_candidates = rank_candidates(job_description, candidates)

    table_data = []
    for candidate, score in ranked_candidates:
        table_data.append([candidate['name'], f"{score * 100:.2f}%"])

    headers = ["Candidate name", "Percentage to suit a job"]
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))


if __name__ == "__main__":
    main()
