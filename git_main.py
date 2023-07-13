import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

GITHUB_API_URL = "https://api.github.com"
SEARCH_USERS_URL = f"{GITHUB_API_URL}/search/users"


def get_repositories(user):
    repos_url = user["repos_url"]
    response = requests.get(repos_url)
    if response.status_code != 200:
        return []

    return response.json()


def get_user_info(candidate):
    repos = get_repositories(candidate)
    repo_descriptions = [repo["description"] if repo["description"] else "" for repo in repos]
    bio = candidate.get("bio", "") if candidate.get("bio") else ""
    return bio + " " + " ".join(repo_descriptions)


def rank_candidates(job_description, candidates):
    if not candidates:
        print("No candidates found.")
        return []

    candidate_info = [get_user_info(candidate) for candidate in candidates]
    texts = [job_description] + candidate_info
    vectorizer = TfidfVectorizer(stop_words="english")
    text_matrix = vectorizer.fit_transform(texts)
    similarity_scores = cosine_similarity(text_matrix[0:1], text_matrix[1:])
    sorted_scores_indices = similarity_scores.argsort()[0][::-1]
    ranked_candidates = [(candidates[i], similarity_scores[0][i]) for i in sorted_scores_indices]
    return ranked_candidates


def search_github_users(query, page):
    params = {
        "q": query,
        "sort": "joined",
        "order": "desc",
        "per_page": 30,
        "page": page
    }
    response = requests.get(SEARCH_USERS_URL, params=params)
    if response.status_code != 200:
        print("Error retrieving data from GitHub API.")
        return []

    data = response.json()
    return data.get("items", [])


def main():
    job_description = "Full-stack web developer with at least 3 years of experience in JavaScript, HTML, CSS, and Python. Knowledge of React.js and Django is a plus."
    language = "Python"
    query = f"language:{language}"
    total_found = 0
    page = 1
    table_data = []

    while total_found < 10:
        candidates = search_github_users(query, page)
        ranked_candidates = rank_candidates(job_description, candidates)

        for candidate, score in ranked_candidates:
            if total_found >= 10:
                break

            username = candidate["login"]
            profile_url = candidate["html_url"]
            bio = candidate.get("bio", "")
            table_data.append([username, bio, profile_url, f"{score * 100:.2f}%"])
            total_found += 1

        page += 1

    headers = ["Username", "Bio", "Profile URL", "Percentage to suit a job"]
    print(tabulate(table_data, headers=headers, tablefmt='pretty'))


if __name__ == "__main__":
    main()
