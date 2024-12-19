import pandas as pd
import numpy as np
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

async def fetch_url(url, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return url, response.text, True  # Success on HTTPS
        except Exception as e:
            # print(f"Attempt {attempt} failed for {url} with error: {e}")
            if attempt == retries:  # On final failure, try HTTP fallback
                http_url = url.replace("https://", "http://") if url.startswith("https://") else url
                try:
                    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                        response = await client.get(http_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        return http_url, response.text, True  # Success on HTTP fallback
                except Exception as e:
                    # print(f"HTTP fallback failed for {url}: {e}")
                    pass
    return url, None, False  # Final failure after all retries

async def fetch_all_urls(urls):
    progress = st.progress(0)  # Initialize the progress bar
    total_urls = len(urls)
    batch_size = 30  # total_urls/10
    results = []

    for i in range(0, total_urls, batch_size):
        batch = urls[i:i + batch_size]
        tasks = [fetch_url(url) for url in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)

        # Update progress bar incrementally
        progress.progress(min((i + batch_size) / total_urls, 1.0))

    progress.empty()  # Clear progress bar after completion
    return results

def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    headers = " ".join([h.get_text(separator=" ", strip=True) for h in soup.find_all(["h1", "h2"])])
    body = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
    text = headers + " " + body  # Combine headers with body text
    return re.sub(r"\s+", " ", text.lower())  # Clean up extra spaces and normalize to lowercase

def compute_relevance_scores(texts, keywords, flexible_search=False):
    """
    Calculate relevance scores based on keyword matches.
    - Full matches use TF-IDF.
    - Flexible matches use regex for lemmatized keywords.

    Parameters:
        texts (list of str): List of parsed text from each URL.
        keywords (list of str): List of keywords to match.
        flexible_search (bool): Whether to enable flexible matching.

    Returns:
        np.array: Normalized relevance scores.
    """
    # Step 1: TF-IDF for full matches
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Identify indices of keywords in the TF-IDF matrix
    full_match_indices = [i for i, term in enumerate(feature_names) if term in keywords]
    full_match_scores = tfidf_matrix[:, full_match_indices].sum(axis=1).A1

    # Step 2: Flexible matching for partial matches (if enabled)
    partial_scores = np.zeros(len(texts))
    if flexible_search:
        keyword_patterns = {
            kw: re.compile(rf"\b{lemmatizer.lemmatize(kw)}\w*\b", re.IGNORECASE) for kw in keywords
        }
        for i, text in enumerate(texts):
            partial_scores[i] = sum(len(pattern.findall(text)) for pattern in keyword_patterns.values())

    # Step 3: Combine scores with weights
    final_scores = full_match_scores + 0.5 * partial_scores  # Give partial matches a lower weight

    # Normalize scores between 0 and 1
    min_score, max_score = final_scores.min(), final_scores.max()
    if max_score > min_score:
        normalized_scores = (final_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = final_scores  # Avoid division by zero for uniform scores

    return normalized_scores

def compute_tfidf_scores(texts, keywords):
    # Create TF-IDF vectorizer with ngram_range=(1, 2)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Extract keyword indices (terms that match the provided keywords)
    keyword_indices = [i for i, term in enumerate(vectorizer.get_feature_names_out()) if term in keywords]

    # Sum TF-IDF scores for each website based on the keywords
    keyword_scores = tfidf_matrix[:, keyword_indices].sum(axis=1).A1

    # Count the missing keywords for each website
    missing_keywords_count = []
    for i, text in enumerate(texts):
        matches = sum([1 for keyword in keywords if keyword in text.lower()])
        missing_keywords_count.append(len(keywords) - matches)

    # Calculate penalty based on missing keywords (more missing = higher penalty)
    penalty = np.array(missing_keywords_count) / len(keywords)

    # Apply penalty to keyword scores (subtract penalty from the raw score)
    adjusted_scores = keyword_scores * (1 - penalty)  # Penalty decreases the score for missing keywords

    # Normalize the scores between 0 and 1
    min_score = np.min(adjusted_scores)
    max_score = np.max(adjusted_scores)
    normalized_scores = (adjusted_scores - min_score) / (max_score - min_score)

    return normalized_scores


def main():
    st.title("Website Keyword Scanner")
    st.write("testing in progress...")
    uploaded_file = st.file_uploader("Upload a file with URLs", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        url_column = st.selectbox("Select the URL column name:", df.columns)

        keywords_input = st.text_input("Enter keywords/phrases (Comma-separated):")
        flexible_search = st.checkbox("Enable Flexible Search",
                                      help="This option will match keywords with their base form, ignoring plural forms and verb tenses, to increase the flexibility of the search.")
        advanced_search = st.checkbox("Enable Advanced Search",
                                      help="This option calculates the relevance score of each website using advanced algorithms. Scanning might take longer.")

        if st.button("Scan"):

            keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]

            if not (url_column and keywords):
                st.error("Please enter URL column and keywords.")
                return
            try:
                c_urls = df[url_column].dropna().tolist()
                urls, invalid_urls = check_url(c_urls)
                if not urls:
                    st.error(f"No valid URLs found in column '{url_column}'.")
                    return
            except Exception:
                st.error(f"Cannot read the URL column '{url_column}'.")
                return

            with st.spinner("Scanning websites... This may take a while."):

                # Initialize lists to hold URL and text content
                texts, valid_urls = [], []

                # Fetch and parse HTML for each valid URL
                results = asyncio.run(fetch_all_urls(urls))

                for url, content, success in results:
                    if success:
                        text = parse_html(content)
                        texts.append(text)
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)

                # Calculate relevance scores (including flexible search if enabled)
                if texts:
                    if advanced_search:
                        relevance_scores = compute_relevance_scores(texts, keywords, flexible_search=flexible_search)
                        data = {
                            "URL": valid_urls,
                            "Keywords Found": [{kw: text.count(kw) for kw in keywords} for text in texts],
                            "Total Matches": [sum(text.count(kw) for kw in keywords) for text in texts],
                            "Relevance Score": relevance_scores.flatten()
                        }
                        result_df = pd.DataFrame(data).sort_values(by='Relevance Score', ascending=False)
                    else:
                        data = {
                            "URL": valid_urls,
                            "Keywords Found": [{kw: text.count(kw) for kw in keywords} for text in texts],
                            "Total Matches": [sum(text.count(kw) for kw in keywords) for text in texts],
                        }
                        result_df = pd.DataFrame(data).sort_values(by='Total Matches', ascending=False)
                    st.write("Scanning complete!")
                    st.dataframe(result_df, column_config={
                        "URL": st.column_config.LinkColumn(width="medium")
                    })

            if invalid_urls:
                st.write("Invalid URLs:")
                inv_urls_df = pd.DataFrame(invalid_urls, columns=["Invalid URL"])
                st.dataframe(inv_urls_df, column_config={
                    "Invalid URL": st.column_config.LinkColumn()
                })


if __name__ == "__main__":
    main()
