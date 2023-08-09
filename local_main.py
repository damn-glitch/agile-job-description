import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import PyPDF2
import pdfplumber
def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text


# Step 1: Create a local database of random people
# This list of dictionaries represents a sample database of candidate profiles.
"""
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
"""
local_database = [
    {
        "id": 13,
        "name": "Alisher Beisembekov",
        "profile_url": "local:/13",
        "description": read_pdf('Alisher Beisembekov CV.pdf'),
    },
    {
        "id": 14,
        "name": "Akbayan",
        "profile_url": "local:/14",
        "description": read_pdf('market\\AkbayanResume - Akbayan Bakytzhanova.pdf'),
    },
    {
        "id": 15,
        "name": "Mukhammed",
        "profile_url": "local:/15",
        "description": read_pdf('market\\CV - Mukhammed-Ali Zholdasbay.pdf'),
    },
    {
        "id": 16,
        "name": "Yernar",
        "profile_url": "local:/16",
        "description": read_pdf('market\\CV Ернар Саден (1) - Yernar Saden.pdf'),
    },
    {
        "id": 17,
        "name": "Alina",
        "profile_url": "local:/17",
        "description": read_pdf('market\\CV_Alina_Nurkabekova_En (2) - Alina N.pdf'),
    },
    {
        "id": 18,
        "name": "Adilet",
        "profile_url": "local:/18",
        "description": read_pdf('market\\Frontend_Adilet_Maksatuly_CV - Hunnid Bands.pdf'),
    },
    {
        "id": 19,
        "name": "Melm",
        "profile_url": "local:/19",
        "description": read_pdf('market\\Go Developer (1) - Melm God.pdf'),
    },
    {
        "id": 20,
        "name": "Daryn",
        "profile_url": "local:/20",
        "description": read_pdf('market\\KenessovDarynCV - daryn kenessov.pdf'),
    },
    {
        "id": 21,
        "name": "Nuray",
        "profile_url": "local:/21",
        "description": read_pdf('market\\Resume N - Nuray Yelubay. Yelubay.pdf'),
    },
    {
        "id": 22,
        "name": "Aidar",
        "profile_url": "local:/22",
        "description": read_pdf('market\\Resume-Aidar-Suleimenov - Айдар Сулейменов.pdf'),
    },
    {
        "id": 23,
        "name": "Sanzhar",
        "profile_url": "local:/23",
        "description": read_pdf('market\\Sanzhar_Abduraimov_Software_Developer - Sanzhar Abduraimov.pdf'),
    },
    {
        "id": 24,
        "name": "Alihan",
        "profile_url": "local:/24",
        "description": read_pdf('market\\Резюме - Alihan Boldubaev.pdf'),
    },
    {
        "id": 25,
        "name": "Indira",
        "profile_url": "local:/25",
        "description": read_pdf('market\\IT PMP, PSPO, PSM_Blockchain PM Djambaeva - Indira Djambaeva (1).pdf'),
    },
    {
        "id": 26,
        "name": "Chelsey",
        "profile_url": "local:/26",
        "description": read_pdf('market\\marketing-manager-resume-example.pdf'),
    }
]

# Function to extract keywords from a text (job description)
def extract_keywords(job_description):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(job_description)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return keywords

# Function to search candidates from the local database
def search_candidates(keywords):
    return local_database

# Function to rank candidates based on the cosine similarity between the job description and each candidate's description
def rank_candidates(job_description, candidates):
    candidate_profiles = [candidate['description'] for candidate in candidates]
    texts = [job_description] + candidate_profiles
    vectorizer = TfidfVectorizer()
    text_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    sorted_scores_indices = similarity_scores.argsort()[0][::-1]
    ranked_candidates = [(candidates[i], similarity_scores[0][i]) for i in sorted_scores_indices]
    return ranked_candidates

# Function to generate a recommendation for a candidate
def generate_recommendation(job_description, candidate_description):
    nlp = spacy.load("en_core_web_sm")
    job_doc = nlp(job_description)
    candidate_doc = nlp(candidate_description)

    job_skills = [token.lemma_ for token in job_doc if token.is_alpha and not token.is_stop]
    candidate_skills = [token.lemma_ for token in candidate_doc if token.is_alpha and not token.is_stop]

    missing_skills = [skill for skill in job_skills if skill not in candidate_skills]

    if missing_skills:
        return f"Consider improving skills in: {', '.join(missing_skills[:3])}"
    else:
        return "Skills match the job description"

# Main function
def main():
    # Read job description from 'job_front.txt' file
    with open('job_front.txt', 'r') as file:
        job_description = file.read().strip()

    keywords = extract_keywords(job_description)
    candidates = search_candidates(keywords)
    ranked_candidates = rank_candidates(job_description, candidates)

    table_data = []
    for candidate, score in ranked_candidates:
        recommendation = generate_recommendation(job_description, candidate['description'])
        table_data.append([candidate['name'], f"{score * 100:.2f}%", recommendation])

    headers = ["Candidate name", "Percentage to suit a job", "Recommendation"]
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))

if __name__ == "__main__":
    main()