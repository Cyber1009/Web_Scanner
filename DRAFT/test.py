import streamlit as st
import pandas as pd
import numpy as np
import subprocess
from test_key import KeywordScanner

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

def run_scrapy_script(url_str, keyword_str):
    scanner = KeywordScanner(url_str, keyword_str)  # Create an instance
    results = scanner.run(url_str, keyword_str)  # Call the run method
    if results:
        result_df = pd.DataFrame.from_dict(results, orient='index',
                                           columns=['URL', 'Keyword Matches', 'Total Count'])
        # ... (rest of your code for processing and displaying results)
        st.dataframe(result_df.head())
        return result_df
    else:
        st.warning("No results found during scan.")

if __name__ == "__main__":
    invalid_urls = []
    st.title("URL Keyword Scanner")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header = 0)
        st.dataframe(df.head())
    url_col = str(st.text_input("Enter the URL:"))
    urls = [check_url(u) for u in df[url_col] if u is not np.nan]

    keywords = st.text_input("Enter keywords and phrases separated by commas:")
    keywords = [keyword.strip().lower() for keyword in keywords.split(",") if keyword.strip()]
    keyword_str = ','.join(keywords)

    # ad_search = st.selectbox("Advanced Search Option (TF-IDF):", ["No", "Yes"])

    if st.button("Start Scan"):
        all_results = []
        for url in urls:
            results = run_scrapy_script(url, keyword_str)
            if results is not None:
                all_results.extend(results)
        if all_results:
            result_df = pd.DataFrame.from_dict(all_results, orient='index',
                                               columns=['URL', 'Keyword Matches', 'Total Count'])
            # ... (further processing and display of results)
            st.dataframe(result_df.head())
        else:
            st.warning("No results found for any URL.")