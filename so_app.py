import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import streamlit as st

from nltk.stem import WordNetLemmatizer
# import nltk
# nltk.download('wordnet')

# Initialize the lemmatizer for stemming/lemmatization
lemmatizer = WordNetLemmatizer()

def check_url(url_list):
    valid_urls = []
    invalid_urls = []
    for u in url_list:
        u = u.strip()
        if u.startswith("www."):
            u = "https://" + u
        if u.startswith(("http://", "https://")):
            valid_urls.append(u)
        else:
            invalid_urls.append(u)
    return valid_urls, invalid_urls

# Function to parse HTML and find keywords
def parse_html_and_links(url, keywords, flexible_search):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
        }
        response = requests.get(url, timeout = 3, headers = headers)
        response.raise_for_status()  # Raise an error if the status code is not 200
        content = response.text

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator=" ", strip=True).lower()

        # Create a dictionary to store keyword counts
        keyword_counts = {}

        for keyword in keywords:
            if flexible_search:

                keyword_lemmatized = " ".join([lemmatizer.lemmatize(word) for word in keyword.split()])
                # Flexible match: check for partial match with stemming
                pattern = r"\b" + re.escape(keyword_lemmatized) + r"\w*\b"
                matches = len(re.findall(pattern, text))
            else:
                # Exact match: count exact occurrences
                matches = content.count(keyword)

            keyword_counts[keyword] = matches

        # Calculate the total match count
        match_count = sum(keyword_counts.values())

        return text, keyword_counts, match_count, True

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        # invalid_urls.append(url)  # Add to invalid URLs list if URL is not accessible
        return "", {}, 0, False


# Function to compute global TF-IDF for all keywords
def compute_global_tfidf(texts, keywords):
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Extract TF-IDF scores for each keyword
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Sum of TF-IDF scores across all documents
    keyword_tfidf = dict(zip(feature_names, tfidf_scores))
    return keyword_tfidf


# Function to calculate relevance score using precomputed global TF-IDF
def calculate_relevance_score(texts, keyword_counts, global_tfidf):
    relevance_scores = []
    for i, text in enumerate(texts):
        # Calculate match count for each keyword
        match_count = sum(keyword_counts[i].values())
        if match_count == 0:
            relevance_scores.append(0.0)
            continue

        # Calculate the local match score (normalized by document length)
        content_length = len(text.split())  # Get the number of words in the content
        normalized_match_score = match_count / content_length if content_length > 0 else 0

        # Calculate the global TF-IDF significance for this page
        tfidf_score = sum(global_tfidf.get(keyword, 0) * keyword_counts[i].get(keyword, 0)
                          for keyword in keyword_counts[i])

        # Final relevance score (normalize and combine match score and global TF-IDF)
        relevance_score = normalized_match_score * tfidf_score
        relevance_scores.append(relevance_score)

    return relevance_scores


# Main function for the Streamlit app
def main():
    st.title("Website Keyword Scanner")
    # st.write("Upload a file with URLs and specify keywords to scan websites.")

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

        # Get URL column name input after other fields
        url_column = st.text_input("Enter the URL column name (case-sensitive):")
        # Validate URL column input
        if url_column:
            if url_column not in df.columns:
                st.error(f"Invalid column name for URLs. Column '{url_column}' not found.")
                return

        # Get keyword input
        keywords_input = st.text_input("Enter keywords/phrases (comma-separated)") # ,"example: technology, innovation")
        if keywords_input:
            keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

        # Flexible search option
        flexible_search = st.checkbox("Enable Flexible Search")
        advanced_search = st.checkbox("Enable Advanced Search")

        # Run the scraper when button is clicked
        if st.button("Scan"):
            if not (url_column and keywords_input):
                st.error("Please enter url column and keywords")
                return
            try:
                # Extract URLs from the specified column
                c_urls = df[url_column].dropna().tolist()
                if c_urls:
                    urls, invalid_urls = check_url(c_urls)
                    if not urls:
                        raise
                        # st.error("Can not find any valid urls in this column")
                        # return
            except:
                st.error("Can not find any valid urls in this column")
                return

            with st.spinner("Scanning websites... This may take a while."):
                valid_urls, texts, keyword_counts, total_matches = [], [], [], []

                for url in urls:
                    # Fetch page content and keyword occurrences
                    text, counts, match_count, success = parse_html_and_links(url, keywords, flexible_search)
                    if success:
                        valid_urls.append(url)
                        texts.append(text)
                        keyword_counts.append(counts)
                        total_matches.append(match_count)
                    else:
                        invalid_urls.append(url)
                        # texts.append('')  # Empty string if URL failed to load
                        # keyword_counts.append({})  # Empty dictionary for failed URLs
                        # total_matches.append(0)  # No matches if URL failed

                # Proceed only if there are valid texts
                if texts:
                    if advanced_search:
                        # Compute global TF-IDF significance across all pages
                        global_tfidf = compute_global_tfidf(texts, keywords)

                        # Calculate relevance scores using global TF-IDF
                        relevance_scores = calculate_relevance_score(texts, keyword_counts, global_tfidf)
                        # Prepare DataFrame for results
                        data = {
                            "URL": valid_urls,
                            "Keyword Occurrences": [str(counts) for counts in keyword_counts],
                            "Match Count": total_matches,
                            "Relevance Score": relevance_scores
                        }
                    else:
                        # Prepare DataFrame for results
                        data = {
                            "URL": valid_urls,
                            "Keywords Found": [str(counts) for counts in keyword_counts],
                            "Total Matches": total_matches
                        }

                    df_result = pd.DataFrame(data)

                    # Display results
                    st.success("Scanning complete!")
                    st.dataframe(df_result)

                else:
                    st.warning("No valid content found for the provided URLs. Please check the URLs and try again.")
                    st.write(invalid_urls)

                # Display invalid URLs
                if invalid_urls:
                    st.warning(f"The following URLs were invalid and could not be processed:")
                    st.write(invalid_urls)



if __name__ == "__main__":
    main()
