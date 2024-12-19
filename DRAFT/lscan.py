import multiprocessing
import streamlit as st
import pandas as pd
import numpy as np
from scrapy.crawler import CrawlerProcess
from lspider import MySpider  # Import spider class
from urllib.parse import urlparse
import concurrent.futures

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

import json

import queue  # Standard library for queue exceptions

def process_results(result_queue, timeout=10):
    """Process results from the queue in parallel with timeout handling."""
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
            keyword_dict = {match['Keyword']: match['Count'] for match in result['Keyword Matches']}
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


def run_scrapy_spider(urls, keywords, result_queue):
    """Function to run Scrapy spider in a separate thread pool."""
    custom_settings = {
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
    }

    # Run Scrapy spider and process results concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        result_processor_future = executor.submit(process_results,
                                                  result_queue)  # Process the results in a separate thread
        futures.append(
            executor.submit(run_scrapy_spider, urls, keywords, result_queue))  # Run Scrapy in a separate thread

        # Wait for Scrapy spider task to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise an exception if something went wrong
            except Exception as e:
                print(f"Error occurred during execution: {e}")

        # Send 'DONE' to result queue to finish processing results
        result_queue.put('DONE')
        result_processor_future.result()  # Ensure result processing completes before proceeding


def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Web Scanner")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            url_column = st.text_input("Enter the URL column name:")
            if url_column not in df.columns:
                st.error("Invalid column name for URLs. Please check your CSV.")
                return
            urls = [check_url(u) for u in (df[url_column].dropna())]  # Extract URLs from the 'URLs' column
            if not urls:
                st.error("No valid URLs found in the CSV.")
                return
        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            return

        keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

        if not keywords:
            st.error("Please enter at least one keyword.")
            return

        if st.button("Start Scan"):
            result_queue = multiprocessing.Queue()

            result_processor = multiprocessing.Process(target=process_results, args=(result_queue,))
            result_processor.start()

            process = multiprocessing.Process(target=run_scrapy_spider, args=(urls, keywords, result_queue))
            process.start()
            st.write("Scrapy is now running in the background...")

            process.join()  # Wait for the Scrapy process to finish
            result_queue.put('DONE')  # Send sentinel value to stop result processing

            result_processor.join()

            result_df = result_queue.get()  # Get the result from the queue

            st.write("Scan completed!")
            if result_df is not None:
                st.dataframe(result_df)
            else:
                st.write("No keyword matches found.")



if __name__ == '__main__':
    app()