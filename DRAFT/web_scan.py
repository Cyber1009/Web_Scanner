import multiprocessing
import streamlit as st
import pandas as pd
import numpy as np
from scrapy.crawler import CrawlerProcess
from web_spider import MySpider  # Import spider class
from urllib.parse import urlparse
import queue
import json
import io

def check_url(url, invalid_urls):
    """Checks the validity of a URL and prepends 'https://' if necessary."""
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url  # Prepend https:// for URLs starting with www.
    if not url.startswith("http") or not urlparse(url).scheme in ['http', 'https']:
        invalid_urls.append(url)  # Add to invalid URLs list
        return None  # Invalid URL
    return url  # Return valid URL

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

def save_invalid_urls(invalid_urls):
    """Save invalid URLs to a file (TXT, CSV, or DataFrame)."""
    if invalid_urls:
        # Save to CSV
        invalid_df = pd.DataFrame({'Invalid URLs': invalid_urls})
        # invalid_df.to_csv("invalid_urls.csv", index=False)
        st.dataframe(invalid_df)

def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Web Scanner")

    uploaded_file = st.file_uploader("Upload your file", type=["csv","xlsx"])

    url_column_error = None
    keywords_error = None
    urls = []
    keywords = []
    invalid_urls = []

    if uploaded_file is not None:
        try:
            # Read file based on its extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            st.dataframe(df.head())  # Display the first few rows of the dataframe

            # Display URL column and keywords inputs below the dataframe
            url_column = st.text_input("Enter the URL column name:")

            # Check if URL column is valid
            if url_column:
                if url_column not in df.columns:
                    url_column_error = "Invalid column name for URLs. Please check your file."
                    st.error(url_column_error)
                else:
                    urls = [check_url(u, invalid_urls) for u in df[url_column].dropna()]
                    # st.error(urls)
                    if not urls:
                        url_column_error = "No valid URLs found in the file."
                        st.error(url_column_error)

            keywords_input = st.text_input("Enter keywords and phrases separated by commas:")

            # Check if keywords input is valid (only validate on scan button click)
            if keywords_input:
                keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

            # Proceed only if inputs are valid
            if not keywords and url_column and st.button("Start Scan"):
                keywords_error = "Please enter at least one keyword."
                st.error(keywords_error)

            if not url_column_error and keywords and st.button("Start Scan"):
                if not urls:
                    st.error("No valid URLs found in the file.")
                    return

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

                result_df = pd.DataFrame(result_queue.get())  # Get the result from the queue

                result_df = result_df.sort_values(by='Total Count', ascending=False)
                st.write("Scan completed!")
                st.dataframe(result_df)
                # if result_df is not None:

                    # st.dataframe(result_df.head())
                    # Save the result to a CSV or Excel file
                    # result_file = "scan_results.csv"
                    # result_df.to_csv(result_file, index=False)

                    # # Provide a download button
                    # with open(result_file, "rb") as file:
                    #     st.download_button(
                    #         label="Download Scan Results",
                    #         data=file,
                    #         file_name=result_file,
                    #         mime="text/csv"
                    #     )

                # else:
                #     st.write("No keyword matches found.")
                # Save invalid URLs after scan
                save_invalid_urls(invalid_urls)
        except Exception as e:
            st.error(f"Unable to read the file: {e}")

if __name__ == '__main__':
    app()
