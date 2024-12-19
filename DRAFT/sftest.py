import pandas as pd
import numpy as np
import httpx
import asyncio
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import streamlit as st
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Semaphore for controlling concurrency in fetch_url
semaphore = asyncio.Semaphore(10)


# Function to validate URLs
async def validate_urls(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.head(url) for url in urls]
        responses = await asyncio.gather(*tasks)
    valid_urls = [url for url, response in zip(urls, responses) if response.status_code == 200]
    return valid_urls


# Function to check and validate URLs
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


# Function to fetch URLs with retries
async def fetch_url(url, retries=2, semaphore=None):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    if semaphore:
        async with semaphore:
            for attempt in range(1, retries + 1):
                try:
                    async with httpx.AsyncClient(follow_redirects=True) as client:
                        response = await client.get(url, headers=headers, timeout=10)
                        response.raise_for_status()
                        return url, response.text, True  # Success on HTTPS
                except Exception as e:
                    if attempt == retries:  # On final failure, try HTTP fallback
                        http_url = url.replace("https://", "http://") if url.startswith("https://") else url
                        try:
                            async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                                response = await client.get(http_url, headers=headers, timeout=10)
                                response.raise_for_status()
                                return http_url, response.text, True  # Success on HTTP fallback
                        except Exception as e:
                            pass
    return url, None, False  # Final failure after all retries


# Function to fetch all URLs concurrently in batches
async def fetch_all_urls(urls, batch_size=30):
    progress = st.progress(0)  # Initialize the progress bar
    total_urls = len(urls)
    results = []

    for i in range(0, total_urls, batch_size):
        batch = urls[i:i + batch_size]
        tasks = [fetch_url(url, semaphore=semaphore) for url in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)

        # Update progress bar incrementally
        progress.progress(min((i + batch_size) / total_urls, 1.0))

    progress.empty()  # Clear progress bar after completion
    return results


# Function to parse HTML content and extract text
def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    # Extract headers and body, clean, and lower in one step
    text = " ".join([h.get_text(separator=" ", strip=True) for h in soup.find_all(["h1", "h2"])] +
                    [soup.body.get_text(separator=" ", strip=True) if soup.body else ""])
    return re.sub(r"\s+", " ", text).lower()  # Clean up extra spaces and normalize to lowercase


# Function to compute relevance scores based on keyword matches
def compute_relevance_scores(texts, keywords, flexible_search=False):
    """
    Calculate relevance scores based on keyword matches.
    - Full matches use TF-IDF.
    - Flexible matches use regex for lemmatized keywords.
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
        # Pre-compile the keyword patterns only once
        keyword_patterns = {kw: re.compile(rf"\b{lemmatizer.lemmatize(kw)}\w*\b", re.IGNORECASE) for kw in keywords}

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


# Function to create a DataFrame with the results
def create_result_dataframe(valid_urls, texts, keywords, relevance_scores=None):
    keyword_data = [{kw: text.count(kw) for kw in keywords} for text in texts]
    total_matches = [sum(text.count(kw) for kw in keywords) for text in texts]

    # Ensure we only create a DataFrame if lengths match
    if len(valid_urls) != len(texts):
        raise ValueError("The lengths of valid_urls and texts do not match.")

    data = {
        "URL": valid_urls,
        "Keywords Found": keyword_data,
        "Total Matches": total_matches
    }

    if relevance_scores is not None:
        if len(valid_urls) != len(relevance_scores):
            raise ValueError("The lengths of valid_urls and relevance_scores do not match.")
        data["Relevance Score"] = relevance_scores.flatten()

    return pd.DataFrame(data).sort_values(by='Relevance Score' if relevance_scores is not None else 'Total Matches',
                                          ascending=False)


# Main function for Streamlit UI and logic
def main():
    st.title("Website Keyword Scanner")
    uploaded_file = st.file_uploader("Upload a file with URLs", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        url_column = st.selectbox("Select the URL column name:", df.columns)
        keywords_input = st.text_input("Enter keywords/phrases (Comma-separated):")
        flexible_search = st.checkbox("Enable Flexible Search")

        if st.button("Scan"):
            keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
            if not keywords:
                st.error("Please enter some keywords.")
                return

            # Process URLs and fetch content
            valid_urls, invalid_urls = check_url(df[url_column].dropna().tolist())
            if not valid_urls:
                st.error(f"No valid URLs found in column '{url_column}'.")
                return

            # Fetch and parse HTML content for each valid URL
            with st.spinner("Scanning websites... This may take a while."):
                results = asyncio.run(fetch_all_urls(valid_urls))

                # Parse content and compute relevance scores
                texts = []
                for url, content, success in results:
                    if success:
                        text = parse_html(content)
                        texts.append(text)
                    else:
                        invalid_urls.append(url)

                # Calculate relevance scores if there are texts to analyze
                relevance_scores = None
                if texts:
                    relevance_scores = compute_relevance_scores(texts, keywords, flexible_search)
                    result_df = create_result_dataframe(valid_urls[:len(texts)], texts, keywords, relevance_scores)
                    st.write("Scanning complete!")
                    st.dataframe(result_df)

                # Display invalid URLs
                if invalid_urls:
                    st.write("Invalid URLs:")
                    inv_urls_df = pd.DataFrame(invalid_urls, columns=["Invalid URL"])
                    st.dataframe(inv_urls_df)


if __name__ == "__main__":
    main()
