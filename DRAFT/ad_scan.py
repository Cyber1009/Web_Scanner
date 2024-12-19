import multiprocessing
import streamlit as st
import pandas as pd
import numpy as np
from scrapy.crawler import CrawlerProcess

from urllib.parse import urlparse
import queue
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')

from ad_spider import MySpider  # Import spider class

def check_url(url):
    """Checks the validity of a URL and prepends 'https://' if necessary."""
    url = url.strip()
    parsed_url = urlparse(url)
    if parsed_url.scheme == '':
        url = 'https://' + url
    if not urlparse(url).scheme in ['http', 'https']:
        print(f"Invalid URL: {url}. Skipping...")
        return np.nan
    return url

def run_scrapy_spider(urls, keywords, result_queue):
    """Run Scrapy spider and collect results."""
    process = CrawlerProcess(settings={
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'DOWNLOAD_TIMEOUT': 10,
        'CONCURRENT_REQUESTS': 10,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 2,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'CLOSESPIDER_PAGECOUNT': 100,
        'ROBOTSTXT_OBEY': False,
    })

    # Pass the URLs, keywords, and result queue to the spider
    process.crawl(MySpider, urls=urls, keywords=keywords, result_queue=result_queue)  # Pass class, not instance
    process.start()  # Start the Scrapy process

def process_results(result_queue, timeout=10):
    """Process results from the queue."""
    scan_results = []
    while True:
        try:
            result = result_queue.get(timeout=timeout)
            if result == 'DONE':
                break  # Sentinel value to end the processing loop
            scan_results.append(result)
        except queue.Empty:
            print("Queue is empty or timeout reached, processing incomplete.")
            break

    if scan_results:
        # Consolidate results into one row per URL
        consolidated_results = []
        for result in scan_results:
            # Ensure every URL has 'Keyword Matches' and 'Total Count' columns
            keyword_dict = {match['Keyword']: match['Count'] for match in result.get('Keyword Matches', [])}
            consolidated_results.append({
                'URL': result['URL'],
                'Keyword Matches': json.dumps(keyword_dict),  # Store as JSON string
                'Total Count': result['Total Count'],
            })

        try:
            result_df = pd.DataFrame(consolidated_results)
            result_queue.put(result_df)  # Send the DataFrame back to the main process
        except Exception as e:
            print(f"Error processing results: {e}")
            result_queue.put(None)

def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Web Scanner")

    uploaded_file = st.file_uploader("Upload your file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            return

        try:
            url_column = st.text_input("Enter the URL column name:")
            if url_column not in df.columns:
                st.error("Invalid column name for URLs. Please check your file.")
                return
            urls = [check_url(u) for u in (df[url_column].dropna())]  # Extract URLs from the 'URLs' column
            if not urls:
                st.error("No valid URLs found in the file.")
                return
        except Exception as e:
            st.error("Unable to find urls")
            return

        keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

        if not keywords:
            st.error("Please enter at least one keyword.")
            return

        advanced_search = st.checkbox("Enable advanced search (slower)")

        if st.button("Start Scan"):
            result_queue = multiprocessing.Queue()

            # Process to handle result processing
            result_processor = multiprocessing.Process(target=process_results, args=(result_queue,))
            result_processor.start()

            # Process to run Scrapy spider
            process = multiprocessing.Process(target=run_scrapy_spider, args=(urls, keywords, result_queue))
            process.start()
            st.write("Scrapy is now running in the background...")

            process.join()  # Wait for the Scrapy process to finish
            result_queue.put('DONE')  # Send sentinel value to stop result processing

            result_processor.join()

            result_df = result_queue.get()  # Get the result from the queue

            if isinstance(result_df, pd.DataFrame):  # Ensure it's a DataFrame
                st.write("Scan completed!")

                # Apply TF-IDF analysis if advanced search is enabled
                if advanced_search:
                    if not result_df.empty:
                        # Filter rows with valid content
                        valid_results = result_df[result_df['Keyword Matches'].notna()]
                        contents = [content for content in valid_results['Keyword Matches']]  # Get content for TF-IDF

                        stop_words = stopwords.words('english')  # Use English stop words
                        tfidf_scores = []

                        # Create a TfidfVectorizer
                        vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, smooth_idf=True)

                        # Apply TF-IDF transformation to the content
                        tfidf_matrix = vectorizer.fit_transform(contents)

                        # Calculate total TF-IDF relevance score for keywords
                        for i, content in enumerate(contents):
                            # Extract the TF-IDF values for the keywords in the content
                            keyword_indices = [vectorizer.vocabulary_.get(keyword) for keyword in keywords if
                                               keyword in vectorizer.vocabulary_]

                            if keyword_indices:
                                # Sum of the TF-IDF scores of the matched keywords
                                keyword_tfidf = tfidf_matrix[i, keyword_indices].sum()

                                # Normalize by the number of matched keywords
                                keyword_count = len(keyword_indices)
                                normalized_tfidf = keyword_tfidf / keyword_count if keyword_count > 0 else 0
                            else:
                                normalized_tfidf = 0  # If no keywords match, set relevance to 0

                            tfidf_scores.append(normalized_tfidf)

                        # Add relevance scores to the dataframe
                        result_df["Relevance"] = tfidf_scores

                        # Sort by relevance
                        result_df = result_df.sort_values(by='Relevance', ascending=False)
                else:
                    result_df = result_df.sort_values(by='Total Count', ascending=False)
                st.dataframe(result_df)
            else:
                st.write("No keyword matches found.")

if __name__ == '__main__':
    app()
