from collections import Counter
import numpy as np
import pandas as pd
import asyncio
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import requests
import re


# Function to validate and clean URLs
def is_valid_url(url):
    url = url.strip()
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Function to parse the HTML content and find the matching keywords
def parse_html_and_links(url, keywords, depth):
    """Retrieve HTML content, and return keywords with occurrences."""
    print('url: ', url)
    try:
        response = requests.get(url, headers = {"User-Agent":"Mozilla/5.0"})
        print('re:', url, response)
        if response.status_code != 200:
            return "", {}, 0, False  # Added False to indicate failure
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text().lower()

        # Count occurrences of each keyword/phrase
        keyword_counts = {kw: len(re.findall(r"\b" + re.escape(kw.lower()) + r"\b", text)) for kw in keywords}

        # Calculate the total match count (sum of all keyword occurrences)
        total_matches = sum(keyword_counts.values())

        return text, keyword_counts, total_matches, True  # True indicates success

    except Exception as e:
        return "", {}, 0, False


# Calculate TF-IDF relevance score based on matched keywords
def calculate_relevance_score(texts, keywords):
    """Calculate relevance score for each website based on the TF-IDF of keywords."""
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate relevance score as the sum of TF-IDF scores for matched keywords
    relevance_scores = tfidf_matrix.sum(axis=1).A1  # A1 converts to a dense array
    return relevance_scores


# Format the results into a user-friendly structure
def main():
    st.title("Website Keyword Scanner")
    st.write("Upload a file with URLs and specify keywords to scan websites.")

    # File upload
    uploaded_file = st.file_uploader("Upload a file with URLs (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file:
        # Load URLs from the uploaded file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display the first few rows to help the user identify the URL column
        st.dataframe(df.head())

        # Get URL column name input
        url_column = st.text_input("Enter the URL column name:")

        # Validate URL column input
        if not url_column:
            st.error("Please enter a valid column name for URLs.")
            return

        if url_column not in df.columns:
            st.error(f"Invalid column name for URLs. Column '{url_column}' not found.")
            return

        # Extract URLs from the specified column
        urls = df[url_column].dropna().tolist()


        # Keywords and depth input
        keywords_input = st.text_area("Enter keywords/phrases (comma-separated)", "example, technology, innovation")
        keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
        depth = st.slider("Subpage Depth", 0, 3, 1)

        # Run the scraper when button is clicked
        if st.button("Scan"):
            with st.spinner("Scanning websites... This may take a while."):
                texts, keyword_counts, total_matches, relevance_scores = [], [], [], []
                invalid_urls = []

                # Scan each URL
                for url in urls:

                    print('1:', url)

                    # Check if URL is valid before processing
                    if not is_valid_url(url):
                        invalid_urls.append(url)  # Add to invalid URLs list
                        # Add placeholders for invalid URLs
                        texts.append('')
                        keyword_counts.append({})
                        total_matches.append(0)
                        relevance_scores.append(0.0)
                        continue  # Skip invalid URLs

                    # Fetch page content and keyword occurrences
                    text, counts, match_count, success = parse_html_and_links(url, keywords, depth)
                    print('2:', success)

                    if success:
                        texts.append(text)
                        keyword_counts.append(counts)
                        total_matches.append(match_count)
                        relevance_scores.append(0.0)  # Placeholder for relevance score
                    else:
                        texts.append('')  # Empty string if URL failed to load
                        keyword_counts.append({})  # Empty dictionary for failed URLs
                        total_matches.append(0)  # No matches if URL failed
                        relevance_scores.append(0.0)  # No relevance score if URL failed

                # Calculate relevance scores (TF-IDF)
                if texts:
                    relevance_scores = calculate_relevance_score(texts, keywords)

                # Prepare DataFrame for results
                data = {
                    "URL": urls,
                    "Keyword Occurrences": [str(counts) for counts in keyword_counts],
                    "Match Count": total_matches,
                    "Relevance Score": relevance_scores
                }

                # Create DataFrame only with valid URLs
                df_result = pd.DataFrame(data)

                # Display results
                st.success("Scanning complete!")
                st.dataframe(df_result)

                # Display invalid URLs
                if invalid_urls:
                    st.warning(f"The following URLs were invalid and could not be processed:")
                    st.write(invalid_urls)

                # Provide download link for results
                st.download_button(
                    "Download Results",
                    data=df_result.to_csv(index=False),
                    file_name="scanned_results.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()