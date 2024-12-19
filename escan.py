import streamlit as st
import pandas as pd
from scrapy.crawler import CrawlerRunner
from twisted.internet import defer, reactor
from twisted.internet.task import react
from espider import MySpider  # Import your spider
from urllib.parse import urlparse
from scrapy.signalmanager import dispatcher
from scrapy import signals


def check_url(url, invalid_urls):
    """Checks the validity of a URL and prepends 'https://' if necessary."""
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url  # Prepend https:// for URLs starting with www.
    if not url.startswith("http") or not urlparse(url).scheme in ['http', 'https']:
        if url not in invalid_urls:
            invalid_urls.append(url)  # Add to invalid URLs list
        return None  # Invalid URL
    return url  # Return valid URL


def save_invalid_urls(invalid_urls):
    """Displays and allows downloading of invalid URLs."""
    if invalid_urls:
        st.write("Invalid URLs:")
        invalid_df = pd.DataFrame({'Invalid URLs': invalid_urls})
        st.dataframe(invalid_df)
        st.download_button(
            label="Download Invalid URLs",
            data=invalid_df.to_csv(index=False),
            file_name="invalid_urls.csv",
            mime="text/csv"
        )


@defer.inlineCallbacks
def run_scrapy_spider(urls, keywords, invalid_urls):
    runner = CrawlerRunner(settings={
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

    results = []

    def collect_results(item):
        results.append(item)  # Store each result

    dispatcher.connect(collect_results, signal=signals.item_scraped)

    # Run the spider
    yield runner.crawl(MySpider, urls=urls, keywords=keywords, invalid_urls=invalid_urls)
    defer.returnValue(results)


def run_scrapy_with_reactor(urls, keywords, invalid_urls):
    """Run the Scrapy crawler and wait for completion."""
    return react(lambda reactor: run_scrapy_spider(urls, keywords, invalid_urls))


def app():
    """Streamlit app to upload CSV and perform scanning."""
    st.title("Web Scanner")

    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])
    urls = []
    keywords = []
    invalid_urls = []

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.dataframe(df.head())

            url_column = st.text_input("Enter the URL column name:")
            if url_column and url_column not in df.columns:
                st.error("Invalid column name for URLs. Please check your file.")

            keywords_input = st.text_input("Enter keywords and phrases separated by commas:")
            if keywords_input:
                keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]

            if st.button("Start Scan"):
                if not url_column or not keywords:
                    st.error("Please provide both a valid URL column and keywords.")
                    return

                urls = [check_url(u, invalid_urls) for u in df[url_column].dropna()]
                if not urls:
                    st.error("No valid URLs found.")
                    return

                with st.spinner("Scanning in progress..."):
                    try:
                        results = run_scrapy_with_reactor(urls, keywords, invalid_urls)
                        save_invalid_urls(invalid_urls)

                        if results:
                            result_df = pd.DataFrame(results)
                            result_df = result_df.sort_values(by='Total Count', ascending=False)
                            st.write("Scan completed!")
                            st.dataframe(result_df)
                        else:
                            st.write("No matches found.")
                    except Exception as e:
                        st.error(f"Error during scanning: {e}")
        except Exception as e:
            st.error(f"Unable to read the file: {e}")


if __name__ == "__main__":
    app()
