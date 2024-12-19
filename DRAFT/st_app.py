import streamlit as st
import pandas as pd
import numpy as np
from keyword_scanner import KeywordScanner
import subprocess
from scrapy.crawler import CrawlerProcess

# def run_scrapy_script(url_str, keyword_str):
#     process = CrawlerProcess()
#     subprocess.run(["python", "keyword_scanner.py", url_str, keyword_str])
#     # Collect results from the crawler
#     scan_result = []
#
#     for crawler in process.crawlers:
#         for item in crawler.spider.item_passer.items():
#             scan_result.append(item)
#             st.write(scan_result)
#     return scan_result


def run_scrapy_script(url_str, keyword_str):
    subprocess.run(["python", "keyword_scanner.py", url_str, keyword_str])
    # Collect results from the crawler
    scan_result = []

    for crawler in process.crawlers:
        for item in crawler.spider.item_passer.items():
            scan_result.append(item)

    return scan_result

def check_url(url):
  url = url.strip()
  if url.startswith("www."):
      url = "https://" + url
  if not url.startswith("http"):
      print(f"Invalid URL: {url}. Skipping...")
      invalid_urls.append(url)
      return np.nan
  else:
      return url

if __name__ == "__main__":
    invalid_urls = []
    st.title("URL Keyword Scanner")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header = 0)
            st.dataframe(df.head())
            url_col = str(st.text_input("Enter the URL:"))
            urls = [check_url(u) for u in df[url_col] if u is not np.nan]
            url_str = ','.join(urls)

        except Exception as e:
            st.error(f"Unable to read urls from file: {e}")
            exit()

    keywords = st.text_input("Enter keywords and phrases separated by commas:")
    keywords = [keyword.strip().lower() for keyword in keywords.split(",") if keyword.strip()]
    keyword_str = ','.join(keywords)

    ad_search = st.selectbox("Advanced Search Option (TF-IDF):", ["No", "Yes"])

    if st.button("Start Scan"):
        scan_result = run_scrapy_script(url_str, keyword_str)
        st.write("Scanning in progress...")

        # scan_result = []  # Replace with actual return value from keyword_scanner.parse

        if scan_result:
            result_df = pd.DataFrame.from_dict(scan_result, orient='index', columns=['URL', 'Keyword Matches', 'Total Count'])
            if ad_search == "Yes":
                result_df = KeywordScanner.advanced_search(result_df, keywords)  # Call advanced search function from keyword_scanner.py

            result_df.to_csv("result.scv", index=False)
            st.dataframe(result_df.head())
        else:
            st.warning("No results found during scan.")