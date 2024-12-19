import multiprocessing
import streamlit as st
import pandas as pd
import numpy as np
from scrapy.crawler import CrawlerProcess
from tspider import MySpider  # Import spider class

def check_url(url):
    """Checks the validity of a URL and prepends 'https://' if necessary."""
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url
    if not url.startswith("http"):
        print(f"Invalid URL: {url}. Skipping...")
        return np.nan
    else:
        return url

def process_results(result_queue):
    """Process results from the queue in parallel."""
    scan_results = []
    while True:
        result = result_queue.get()
        if result == 'DONE':
            break  # Sentinel value to end the processing loop
        scan_results.append(result)

    if scan_results:
        # Consolidate results into one row per URL
        consolidated_results = []
        for result in scan_results:
            keyword_dict = {match['Keyword']: match['Count'] for match in result['Keyword Matches']}
            consolidated_results.append({
                'URL': result['URL'],
                'Keyword Matches': str(keyword_dict),  # Store as a stringified dictionary
                'Total Count': result['Total Count'],
            })
        result_df = pd.DataFrame(consolidated_results)
        result_queue.put(result_df)  # Send the DataFrame back to the main process




def run_scrapy_spider(urls, keywords, result_queue):
    """Function to run Scrapy spider in a separate process."""
    # Custom Scrapy settings
    custom_settings = {
        'RETRY_TIMES': 2,  # Limit retries to 3 times (default is 2)
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],  # Retry on these error codes
        'DOWNLOAD_TIMEOUT': 10,  # Set a shorter download timeout
        'CONCURRENT_REQUESTS': 20,  # Reduce number of concurrent requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,  # Limit requests per domain
        'DOWNLOAD_DELAY': 2,  # Introduce a small delay to reduce strain
        'AUTOTHROTTLE_ENABLED': True,  # Enable AutoThrottle to adjust dynamically
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,  # Target 2 requests per second
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',  # Set logging level to reduce output
        'CLOSESPIDER_PAGECOUNT': 100,  # Limit the number of pages crawled to avoid long runs
        'ROBOTSTXT_OBEY': False,
    }

    process = CrawlerProcess(settings=custom_settings)
    process.crawl(MySpider, urls=urls, keywords=keywords, result_queue=result_queue)
    process.start()


def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Web Scanner")

    # Upload CSV with URLs
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load and display the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            url_column = st.text_input("Enter the URL column name:")
            if url_column not in df.columns:
                st.error("Please enter a valid url column.")
                return
            urls = [check_url(u) for u in (df[url_column].dropna())]  # Extract URLs from the 'URLs' column
            if not urls:
                st.error("No valid URLs found in the CSV.")
                return
        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            return

        # Ask user for keywords and other parameters if needed
        keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

        # Ensure that keywords were provided
        if not keywords:
            st.error("Please enter at least one keyword.")
            return

        if st.button("Start Scan"):
            result_queue = multiprocessing.Queue()  # Create a queue to receive results from the spider

            # Create a process to handle result processing in parallel
            result_processor = multiprocessing.Process(target=process_results, args=(result_queue,))
            result_processor.start()

            # Run Scrapy spider in a separate process
            process = multiprocessing.Process(target=run_scrapy_spider, args=(urls, keywords, result_queue))
            process.start()
            st.write("Scrapy is now running in the background...")

            # Collect results from the queue
            process.join()  # Wait for the Scrapy process to finish
            result_queue.put('DONE')  # Send sentinel value to stop result processing

            # Wait for the result processing to complete
            result_processor.join()

            # Retrieve the processed result
            result_df = result_queue.get()  # Get the result from the queue

            # Display the results
            st.write("Scan completed!")
            if result_df is not None:
                st.dataframe(result_df)
            else:
                st.write("No keyword matches found.")


if __name__ == '__main__':
    app()