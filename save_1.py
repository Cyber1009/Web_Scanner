import pandas as pd
import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import streamlit as st
from nltk.stem import WordNetLemmatizer
import asyncio
from urllib.parse import urlparse

lemmatizer = WordNetLemmatizer()

def check_url(url_list):
    valid_urls = []
    invalid_urls = []
    for u in url_list:
        uc = u.strip()
        if uc.startswith("www."):
            uc = "https://" + uc
        if not uc.startswith(("http://", "https://")):
            invalid_urls.append(u)
            continue
        parsed = urlparse(uc)
        if parsed.scheme and parsed.netloc:
            valid_urls.append(uc)
        else:
            invalid_urls.append(u)
    return valid_urls, invalid_urls

async def fetch_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return url, response.text, True  # Successful case
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code} for URL {url}: {e}")
        return url, None, False
    except httpx.RequestError as e:
        print(f"Request error for URL {url}: {e}")
        return url, None, False
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
        return url, None, False

async def fetch_all_urls(urls):
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if result is not None]

def parse_html_and_find_keywords(content, keywords, flexible_search):
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=" ", strip=True).lower()

    # Additional cleaning for better accuracy
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space

    keyword_counts = {}
    for keyword in keywords:
        if flexible_search:
            keyword_lemmatized = " ".join([lemmatizer.lemmatize(word) for word in keyword.split()])
            pattern = r"\b" + re.escape(keyword_lemmatized) + r"\w*\b"
            matches = len(re.findall(pattern, text))
        else:
            matches = text.count(keyword.lower())  # Ensure case-insensitivity
        keyword_counts[keyword] = matches

    match_count = sum(keyword_counts.values())
    return text, keyword_counts, match_count

def compute_global_tfidf(texts, keywords):
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    return dict(zip(feature_names, tfidf_scores))

def calculate_relevance_score(texts, keyword_counts, global_tfidf):
    relevance_scores = []
    for i, text in enumerate(texts):
        match_count = sum(keyword_counts[i].values())
        if match_count == 0:
            relevance_scores.append(0.0)
            continue
        content_length = len(text.split())
        normalized_match_score = match_count / content_length if content_length > 0 else 0
        tfidf_score = sum(global_tfidf.get(keyword, 0) * keyword_counts[i].get(keyword, 0) for keyword in keyword_counts[i])
        relevance_scores.append(normalized_match_score * tfidf_score)
    return relevance_scores

def main():
    st.title("Website Keyword Scanner")
    uploaded_file = st.file_uploader("Upload a file with URLs (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        url_column = st.text_input("Enter the URL column name (case-sensitive):")
        keywords_input = st.text_input("Enter keywords/phrases (comma-separated)")
        flexible_search = st.checkbox("Enable Flexible Search")
        advanced_search = st.checkbox("Enable Advanced Search")

        if st.button("Scan"):
            if not (url_column and keywords_input):
                st.error("Please enter URL column and keywords.")
                return
            try:
                c_urls = df[url_column].dropna().tolist()
                urls, invalid_urls = check_url(c_urls)
                if not urls:
                    st.error("No valid URLs found.")
                    return
            except:
                st.error("Invalid URL column.")
                return

            with st.spinner("Scanning websites... This may take a while."):
                results = asyncio.run(fetch_all_urls(urls))
                texts, keyword_counts, total_matches = [], [], []
                valid_urls = []

                for url, content, success in results:
                    if success:
                        text, counts, match_count = parse_html_and_find_keywords(content, keywords_input.split(','), flexible_search)
                        texts.append(text)
                        keyword_counts.append(counts)
                        total_matches.append(match_count)
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)

                if advanced_search and texts:
                    global_tfidf = compute_global_tfidf(texts, keywords_input.split(','))
                    relevance_scores = calculate_relevance_score(texts, keyword_counts, global_tfidf)
                    data = {
                        "URL": valid_urls,
                        "Keyword Occurrences": [str(counts) for counts in keyword_counts],
                        "Match Count": total_matches,
                        "Relevance Score": relevance_scores
                    }
                else:
                    data = {
                        "URL": valid_urls,
                        "Keywords Found": [str(counts) for counts in keyword_counts],
                        "Total Matches": total_matches
                    }

                st.success("Scanning complete!")
                st.dataframe(pd.DataFrame(data))

            if invalid_urls:
                st.warning("Some URLs could not be processed:")
                st.write(invalid_urls)

if __name__ == "__main__":
    main()
