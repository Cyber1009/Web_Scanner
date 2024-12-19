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
            print(f"Attempt {attempt} failed for {url} with error: {e}")
            if attempt == retries:  # On final failure, try HTTP fallback
                http_url = url.replace("https://", "http://") if url.startswith("https://") else url
                try:
                    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                        response = await client.get(http_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        return http_url, response.text, True  # Success on HTTP fallback
                except Exception as e:
                    print(f"HTTP fallback failed for {url}: {e}")
    return url, None, False  # Final failure after all retries

async def fetch_all_urls(urls, batch_size=10):
    progress = st.progress(0)  # Initialize the progress bar
    total_urls = len(urls)
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
    body = soup.body
    if body:
        text = body.get_text(separator=" ", strip=True).lower()
    else:
        text = ""
    return re.sub(r"\s+", " ", text)  # Clean up extra spaces

def flexible_match(text, keywords):
    """
    Perform flexible matching of keywords in the text.
    Keywords are lemmatized and matched with their base forms using precompiled regex patterns.
    """
    # Preprocess keywords to generate lemmatized regex patterns
    lemmatized_patterns = {
        keyword: re.compile(
            rf"\b{' '.join([lemmatizer.lemmatize(word) for word in keyword.split()])}\w*\b",
            re.IGNORECASE
        )
        for keyword in keywords
    }

    # Count matches for each keyword using its compiled regex pattern
    keyword_counts = {}
    for keyword, pattern in lemmatized_patterns.items():
        keyword_counts[keyword] = len(pattern.findall(text))

    return keyword_counts


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
    uploaded_file = st.file_uploader("Upload a file with URLs", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        url_column = st.selectbox("Select the URL column name:", df.columns)

        keywords_input = st.text_input("Enter keywords/phrases (Comma-separated):")
        flexible_search = st.checkbox("Enable Flexible Search", help="This option will match keywords with their base form, ignoring plural forms and verb tenses, to increase the flexibility of the search.")
        advanced_search = st.checkbox("Enable Advanced Search", help="This option calculates the relevance score of each website using advanced algorithms. Scanning might take longer.")

        if st.button("Scan"):
            if not (url_column and keywords_input):
                st.error("Please enter URL column and keywords.")
                return
            try:
                c_urls = df[url_column].dropna().tolist()
                urls, invalid_urls = check_url(c_urls)
                if not urls:
                    st.error(f"No valid URLs found in column '{url_column}'.")
                    return
            except:
                st.error(f"Can not read the URL column '{url_column}'.")
                return

            with st.spinner("Scanning websites... This may take a while."):

                # Get keywords and lemmatize if flexible search is enabled
                keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]

                # Initialize lists to hold URL and text content
                texts, valid_urls, all_keyword_counts = [], [], []

                # Fetch and parse HTML for each valid URL
                results = asyncio.run(fetch_all_urls(urls))

                for url, content, success in results:
                    if success:
                        text = parse_html(content)
                        texts.append(text)
                        valid_urls.append(url)
                        if flexible_search:
                            # Perform flexible match if enabled
                            all_keyword_counts.append(flexible_match(text, keywords))
                        else:
                            # Direct count of exact keyword matches
                            all_keyword_counts.append({kw: text.count(kw) for kw in keywords})
                    else:
                        invalid_urls.append(url)

                # If advanced search is enabled, calculate TF-IDF relevance scores
                if advanced_search and texts:
                    relevance_scores = compute_tfidf_scores(texts, keywords)
                    data = {
                        "URL": valid_urls,
                        "Keywords Found": [str(counts) for counts in all_keyword_counts],
                        "Total Matches": [sum(counts.values()) for counts in all_keyword_counts],
                        "Relevance Score": relevance_scores.flatten()
                    }
                    # print(data)
                    # for key, value in data.items():
                    #     print(f"Column '{key}' has shape: {len(value)}")
                    result_df = pd.DataFrame(data).sort_values(by='Relevance Score', ascending=False)

                else:
                    data = {
                        "URL": valid_urls,
                        "Keywords Found": [str(counts) for counts in all_keyword_counts],
                        "Total Matches": [sum(counts.values()) for counts in all_keyword_counts]
                    }
                    result_df = pd.DataFrame(data).sort_values(by='Total Matches', ascending=False)

                # st.success()
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
