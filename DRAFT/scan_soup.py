import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from collections import Counter


def fetch_keywords_from_url(url, keywords):
    """Fetches content from a URL and counts keyword occurrences."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text from all paragraphs, headings, etc.
    text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])

    # Count keyword occurrences
    keyword_counts = Counter({keyword: text.lower().count(keyword.lower()) for keyword in keywords})
    total_count = sum(keyword_counts.values())

    return {
        "URL": url,
        "Keyword Matches": keyword_counts,
        "Total Count": total_count,
        "Text": text  # Optionally, save full text content for TF-IDF
    }


def run_scanner(df, url_column, keywords):
    """Run keyword scan for each URL in the dataframe."""
    results = []

    for url in df[url_column].dropna():
        results.append(fetch_keywords_from_url(url, keywords))

    # Convert to DataFrame and display/save results
    result_df = pd.DataFrame(results)
    result_df.to_csv("scan_results.csv", index=False)
    st.dataframe(result_df)


def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Simple URL Keyword Scanner")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            return

        url_column = st.text_input("Enter the URL column name:")
        keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

        if st.button("Start Scan"):
            run_scanner(df, url_column, keywords)


if __name__ == '__main__':
    app()
