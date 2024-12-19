import streamlit as st
import pandas as pd
import numpy as np
import subprocess

from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import defer, reactor
from twisted.internet.threads import deferToThread
from scrapy.signalmanager import dispatcher
from scrapy import signals
from myspider import MySpider  # Import spider class


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


def collect_results(item, results):
    """Function to collect scraped items."""
    results.append(item)


def stop_reactor(_):
    """Stop the reactor once the crawl is finished."""
    reactor.stop()



def run_scrapy_spider(urls, keywords, crawl_subpages, max_subpages):
    """Function to run Scrapy spider."""
    # Scrapy command
    command = [
        'scrapy', 'crawl', 'my_spider',
        '--urls', ','.join(urls),  # Pass the URLs as an argument
        '--keywords', ','.join(keywords),
        '--crawl_subpages', str(crawl_subpages),
        '--max_subpages', str(max_subpages)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout



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
                st.error("Please enter a valid URL column.")
                return
            urls = [check_url(u) for u in (df[url_column].dropna())]  # Extract URLs from the 'URLs' column

            if not urls:
                st.error("No valid URLs found in the CSV.")
                return
        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            return

        # Ask user for keywords and other parameters
        keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords_input.split(",") if keyword.strip()]

        # Ensure that keywords were provided
        if not keywords:
            st.error("Please enter at least one keyword.")
            return

        # Ask user if they want to crawl subpages
        crawl_subpages = st.checkbox("Crawl subpages?", value=True)
        max_subpages = st.slider("Limit the number of subpages to crawl per page:", 1, 20, 5)

        if st.button("Start Scan"):
            st.write("Scrapy is now running in the background...")

            # Run Scrapy spider and collect results
            results = run_scrapy_spider(urls, keywords, crawl_subpages, max_subpages)

            # Display the results
            if results:
                st.write("Results:", results)  # This will show the results in the Streamlit app
                result_df = pd.DataFrame(results)
                st.write("Scan Results:")
                st.dataframe(result_df)
            else:
                st.write("No keyword matches found.")


if __name__ == '__main__':
    app()
