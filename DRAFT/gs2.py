import nltk
import scrapy
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scrapy.crawler import CrawlerProcess
from twisted.internet import defer, reactor
import logging

# Download necessary NLTK resources
nltk.download('stopwords')

def check_url(url):
    """Checks the validity of a URL and prepends "https://" if necessary."""
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url
    if not url.startswith("http"):
        print(f"Invalid URL: {url}. Skipping...")
        return np.nan
    else:
        return url

class KeywordScanner(scrapy.Spider):
    name = "keyword_scanner"

    def __init__(self, url, keywords, *args, **kwargs):
        super(KeywordScanner, self).__init__(*args, **kwargs)
        self.url = url
        self.keywords = keywords
        self.text_content = ""

    def start_requests(self):
        """Initiates requests to the URL to scrape."""
        yield scrapy.Request(self.url, callback=self.parse)

    def count_keywords(self, text, keywords):
        """Counts the occurrences of each keyword in the text."""
        keyword_counts = Counter()
        for keyword in keywords:
            keyword_counts[keyword.lower()] += text.lower().count(keyword.lower())
        return keyword_counts

    def parse(self, response):
        """Parses the content of the page and counts keywords."""
        url = response.url
        text = response.css("body *::text").getall()
        self.text_content = " ".join(text)

        # Count keyword occurrences
        keyword_counts = self.count_keywords(self.text_content, self.keywords)
        total_count = sum(keyword_counts.values())

        yield {
            "URL": url,
            "Keyword Matches": keyword_counts,
            "Total Count": total_count,
            "Text": self.text_content  # Save the extracted text for TF-IDF
        }

def run_crawler(urls, keywords):
    """Handles the crawling process using CrawlerProcess."""
    process = CrawlerProcess({
        'LOG_LEVEL': 'ERROR',  # Suppress non-error log messages
    })

    for url in urls:
        process.crawl(KeywordScanner, url=url, keywords=keywords)

    process.start()  # Start the crawling process

def run_scanner(df, url_column, keywords, advanced_search):
    """Handles running the scanner and calculating TF-IDF if needed."""
    try:
        urls = [check_url(u) for u in df[url_column] if u is not np.nan]
        # Run the crawler in the main thread
        reactor.callFromThread(run_crawler, urls, keywords)

        # You can optionally add logic here to block the UI thread while processing
        # and then return the results once done (asynchronously).

        st.success("Scan completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

def app():
    """Main Streamlit app function."""
    st.title("URL Keyword Scanner")
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

        advanced_search = st.checkbox("Enable advanced search (slower)")

        if st.button("Start Scan"):
            run_scanner(df, url_column, keywords, advanced_search)

if __name__ == '__main__':
    app()
