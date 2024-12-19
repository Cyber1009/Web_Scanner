import multiprocessing
import streamlit as st
import pandas as pd
import numpy as np
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from my_spider import MySpider  # Import spider class

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

def run_scrapy_spider(urls, keywords, result_queue):
    """Function to run Scrapy spider in a separate process."""

    # Custom Scrapy settings
    custom_settings = {
        'RETRY_TIMES': 5,  # Retry attempts
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408],  # Retry on these error codes
        'DOWNLOAD_TIMEOUT': 15,  # Timeout in seconds
        'CONCURRENT_REQUESTS': 16,  # Increase number of concurrent requests
        'DOWNLOAD_DELAY': 1,  # Delay between requests (in seconds)
        'AUTOTHROTTLE_ENABLED': True,  # Enable AutoThrottle to manage delay dynamically
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',# Set user-agent
    }
    process = CrawlerProcess(settings=custom_settings)
    process.crawl(MySpider, urls=urls, keywords=keywords, result_queue=result_queue)  # Pass the queue to the spider
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
            urls = df[url_column].dropna().tolist()  # Extract URLs from the 'URLs' column
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

            # Run Scrapy spider in a separate process
            process = multiprocessing.Process(target=run_scrapy_spider, args=(urls, keywords, result_queue))
            process.start()
            st.write("Scrapy is now running in the background...")

            # Collect results from the queue after the spider finishes
            process.join()  # Wait for the Scrapy process to finish
            st.write("Scan completed!")

            # Now display the results
            scan_results = []
            while not result_queue.empty():
                scan_results.append(result_queue.get())

            if scan_results:
                result_df = pd.DataFrame(scan_results)
                st.write("Scan Results:")
                st.dataframe(result_df)
            else:
                st.write("No keyword matches found.")

if __name__ == '__main__':
    app()
