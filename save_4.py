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


async def fetch_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return url, response.text, True
    except Exception as e:
        if url.startswith("https://"):
            try:
                http_url = url.replace("https://", "http://")
                async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                    response = await client.get(http_url, headers=headers, timeout=10)
                    response.raise_for_status()
                    return url, response.text, True
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return url, None, False
        return url, None, False

async def fetch_all_urls(urls):
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if result is not None]


def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    body = soup.body
    if body:
        text = body.get_text(separator=" ", strip=True).lower()
    else:
        text = ""
    return re.sub(r"\s+", " ", text)  # Clean up extra spaces


def flexible_match(text, keywords):
    keyword_counts = {}
    for keyword in keywords:
        lemmatized_keyword = " ".join([lemmatizer.lemmatize(word) for word in keyword.split()])
        pattern = re.compile(rf"\b{lemmatized_keyword}\w*\b", re.IGNORECASE)
        keyword_counts[keyword] = len(pattern.findall(text))
    return keyword_counts


def compute_tfidf_scores(texts, keywords):
    # Create TF-IDF vectorizer with keywords as the vocabulary and ngram_range=(1, 2)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # print("TF-IDF Matrix for each document:")
    # print(tfidf_matrix.toarray())
    # print("Feature names (terms):")
    # print(vectorizer.get_feature_names_out())

    # Optionally, you can add weights based on keyword importance (post-processing)
    keyword_indices = [i for i, term in enumerate(vectorizer.get_feature_names_out()) if term in keywords]
    keyword_scores = tfidf_matrix[:, keyword_indices].sum(axis=1).A1  # Sum scores of only the keywords

    # Normalize the keyword scores
    normalized_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores))
    return normalized_scores

    # scores = tfidf_matrix.max(axis=1).A1
    # return scores


def main():
    st.title("Website Keyword Scanner")
    uploaded_file = st.file_uploader("Upload a file with URLs", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        url_column = st.text_input("Enter the URL column name (Case-sensitive):")
        if url_column:
            if url_column not in df.columns:
                st.error(f"Invalid column name. Column '{url_column}' not found.")
                return

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
                st.error("Invalid URL column. Please check your file and column name.")
                return

            with st.spinner("Scanning websites... This may take a while."):

                # Get keywords and lemmatize if flexible search is enabled
                keywords = [kw.strip().lower() for kw in keywords_input.split(',')]

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
                        "Relevance Score": relevance_scores
                    }
                    result_df = pd.DataFrame(data).sort_values(by='Relevance Score', ascending=False)

                else:
                    data = {
                        "URL": valid_urls,
                        "Keywords Found": [str(counts) for counts in all_keyword_counts],
                        "Total Matches": [sum(counts.values()) for counts in all_keyword_counts]
                    }
                    result_df = pd.DataFrame(data).sort_values(by='Total Matches', ascending=False)

                st.success("Scanning complete!")
                st.dataframe(result_df, column_config={
                    "URL":st.column_config.LinkColumn(width="medium")
                })

            if invalid_urls:
                st.write("Invalid URLs:")
                inv_urls_df = pd.DataFrame(invalid_urls, columns = ["Invalid URL"])
                st.dataframe(inv_urls_df, column_config={
                    "Invalid URL":st.column_config.LinkColumn()
                })


if __name__ == "__main__":
    main()
