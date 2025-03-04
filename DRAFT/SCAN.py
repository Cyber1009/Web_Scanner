# -*- coding: utf-8 -*-
"""
Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iSPj8-mQRxnRMac8vvl57YhXvVXVpo7Q
"""

#pip install streamlit scrapy scikit-learn

import nltk

nltk.download('stopwords')
import scrapy
import streamlit as st
import pandas as pd
import asyncio
import numpy as np
import keyword_scanner
from keyword_scanner import CrawlerProcess
# from keyword_scanner import Selector
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def check_url(url):
  """
    Checks the validity of a URL and prepends "https://" if necessary.

    Args:
        url (str): The URL to be checked.

    Returns:
        str: The valid URL with "https://" prepended if needed, or None if invalid.
  """
  url = url.strip()
  if url.startswith("www."):
      url = "https://" + url
  if not url.startswith("http"):
      print(f"Invalid URL: {url}. Skipping...")
      # invalid_urls.append(url)
      return np.nan
  else:
      return url

class KeywordScanner(scrapy.Spider):
    name = "keyword_scanner"

    def __init__(self, url, keywords, *args, **kwargs):
        super(KeywordScanner, self).__init__(*args, **kwargs)
        self.url = url
        self.keywords = keywords

    def start_requests(self):
        yield scrapy.Request(self.url, callback=self.parse)

    def count_keywords(self, text, keywords):
        """
        Counts the occurrences of each keyword in the text.

        Args:
            text (str): The text to search.
            keywords (list): A list of keywords to search for.

        Returns:
            dict: A dictionary where keys are keywords and values are their counts.
        """
        keyword_counts = Counter(text.split())
        for keyword in keywords:
            # Consider case-insensitive search (optional)
            keyword_counts[keyword.lower()] += keyword_counts.get(keyword, 0)
        return keyword_counts

    def parse(self, response):
        url = response.url
        text = response.css("body *::text").getall()

        # Combine keywords (phrases and individual words)
        keyword_counts = self.count_keywords(" ".join(text), self.keywords)
        total_count = sum(keyword_counts.values())

        # Follow links if enabled
        if st.session_state.get("advanced_search", False):
            for next_page in response.css("a::attr(href)").getall():
                yield response.follow(next_page, callback=self.parse)

        yield {
            "URL": url,
            "Keyword Matches": keyword_counts,
            "Total Count": total_count,
        }

def main(urls, keywords):
    """
    The main function that orchestrates URL processing, keyword counting,
    and result generation with Scrapy.

    Args:
        urls (list): A list of URLs to process.
        keywords (list): A list of keywords and phrases to search for.

    Returns:
        list: A list of dictionaries containing scan results for each URL.
    """

    results = []
    for url in urls:
        process = CrawlerProcess()
        process.crawl(KeywordScanner, url=url, keywords=keywords)
        process.start()
        for item in process.crawler.items:
            results.append(item)
    return results

def run_scanner(df, url_column, keywords, advanced_search):
    try:
        # df = pd.read_csv(url_file)
        urls = [check_url(u) for u in df[url_column] if u is not np.nan]

        scan_result = asyncio.run(main(urls, keywords))

        # Create a DataFrame from the results
        results_df = pd.DataFrame(scan_result)

        # Calculate TF-IDF scores
        if advanced_search:
            contents = [item['text'] for item in scan_result]
            stop_words = stopwords.words('english')
            tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
            tfidf_matrix = tfidf_vectorizer.fit_transform(contents)

            # ... (Calculate TF-IDF scores for each keyword and add to the DataFrame)

        st.dataframe(results_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    st.title("URL Keyword Scanner")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    # invalid_urls = []

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, header = 0)
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Unable to read the file: {e}")
            exit()

        url_column = st.text_input("Enter the URL column name:")
        # urls = [check_url(u) for u in df[url_column] if u is not np.nan]
        keywords = st.text_input("Enter keywords and phrases separated by commas:")
        keywords = [keyword.strip().lower() for keyword in keywords.split(",") if keyword.strip()]

        advanced_search = st.checkbox("Enable advanced search (slower)")

        if st.button("Start Scan"):
            run_scanner(df, url_column, keywords, advanced_search)

# !pip install colab-xterm
# %load_ext colabxterm
# %xterm