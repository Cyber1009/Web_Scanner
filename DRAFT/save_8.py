import pandas as pd
import streamlit as st
import asyncio
import aiohttp
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
import re
import numpy as np


# Function to check if URLs are valid
def check_url(urls):
    valid_urls = []
    invalid_urls = []
    for url in urls:
        if isinstance(url, str) and re.match(r'^https?://', url):
            valid_urls.append(url)
        elif isinstance(url, str) and url.startswith("www."):
            valid_urls.append("https://" + url)  # Prepend https:// to URLs that start with 'www.'
        else:
            invalid_urls.append(url)
    return valid_urls, invalid_urls


# Function to fetch the HTML content of a URL
async def fetch_url(url, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }
    attempt = 0
    while attempt < retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        return url, content, True
                    else:
                        return url, None, False
        except Exception as e:
            attempt += 1
            time.sleep(2)  # Retry after a delay
    return url, None, False  # Return failure after retries


# Function to extract links from the page to get subpages
def extract_subpage_links(url, content):
    soup = BeautifulSoup(content, "html.parser")
    base_url = urlparse(url).netloc
    links = set()  # Use a set to avoid duplicates
    for anchor in soup.find_all("a", href=True):
        link = anchor["href"]
        # Make sure the link is absolute and belongs to the same domain
        if link.startswith("/"):
            link = urlparse(url)._replace(path=link).geturl()
        elif base_url in link:
            # Validate the link and add it
            links.add(link)
    return links


# Function to fetch subpages along with the main page
async def fetch_subpages(url, retries=2, max_subpages=5):
    """Fetch main page and subpages up to the specified limit."""
    url, content, success = await fetch_url(url, retries)
    if not success or content is None:
        return [], []

    # Parse main page to extract subpage URLs
    subpages = extract_subpage_links(url, content)
    # Limit subpages to the max number specified
    subpages = list(subpages)[:max_subpages]

    # Fetch all subpage URLs
    subpage_results = []
    for subpage in subpages:
        subpage_result = await fetch_url(subpage)
        subpage_results.append(subpage_result)

    # Return both main page and its subpages
    return [(url, content)] + subpage_results  # Return the main page and its subpages


# Function to fetch all URLs (main page + subpages if enabled)
async def fetch_all_urls(urls, scan_subpages=False, max_subpages=5, batch_size=10):
    """Fetch both main URLs and subpages if enabled."""
    progress = st.progress(0)  # Initialize the progress bar
    total_urls = len(urls)
    results = []

    for i in range(0, total_urls, batch_size):
        batch = urls[i:i + batch_size]
        tasks = [fetch_subpages(url, max_subpages=max_subpages) if scan_subpages else fetch_url(url) for url in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)

        # Update progress bar incrementally
        progress.progress(min((i + batch_size) / total_urls, 1.0))

    progress.empty()  # Clear progress bar after completion
    return results


# Function to parse the HTML content (extract text)
def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()


# Function to calculate the relevance score of keywords in the text
def compute_relevance_scores(texts, keywords, flexible_search=False):
    scores = []
    for text in texts:
        score = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text.lower().count(keyword_lower)
            if flexible_search:
                # Simple flexibility by allowing plural/singular form
                count += text.lower().count(keyword_lower + "s")
            score += count
        scores.append(score)
    return np.array(scores)


def main():
    st.title("Website Keyword Scanner")
    uploaded_file = st.file_uploader("Upload a file with URLs", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        url_column = st.selectbox("Select the URL column name:", df.columns)
        keywords_input = st.text_input("Enter keywords/phrases (Comma-separated):")
        flexible_search = st.checkbox("Enable Flexible Search",
                                      help="This option will match keywords with their base form.")
        scan_subpages = st.checkbox("Scan Subpages", help="Enable scanning of subpages linked from the main page.")
        max_subpages = st.slider("Limit number of subpages per URL:", min_value=1, max_value=20, value=5)

        if st.button("Scan"):
            keywords = [kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]
            if not (url_column and keywords):
                st.error("Please enter URL column and keywords.")
                return

            try:
                c_urls = df[url_column].dropna().tolist()
                urls, invalid_urls = check_url(c_urls)
                if not urls:
                    st.error(f"No valid URLs found in column '{url_column}'.")
                    return
            except Exception:
                st.error(f"Cannot read the URL column '{url_column}'.")
                return

            with st.spinner("Scanning websites... This may take a while."):

                # Fetch and parse HTML for each valid URL (including subpages if enabled)
                results = asyncio.run(fetch_all_urls(urls, scan_subpages=scan_subpages, max_subpages=max_subpages))

                texts, valid_urls = [], []
                for result in results:
                    if result:
                        combined_content = ""
                        for url, content, success in result:
                            if success and content:
                                combined_content += parse_html(
                                    content)  # Combine content from the main page and subpages
                                valid_urls.append(url)
                            else:
                                invalid_urls.append(url)

                        texts.append(combined_content)  # Append the combined content for this URL

                # Calculate relevance scores (including flexible search if enabled)
                if texts:
                    if flexible_search:
                        relevance_scores = compute_relevance_scores(texts, keywords, flexible_search=flexible_search)
                        data = {
                            "URL": valid_urls,
                            "Keywords Found": [{kw: combined_text.count(kw) for kw in keywords} for combined_text in
                                               texts],
                            "Total Matches": [sum(combined_text.count(kw) for kw in keywords) for combined_text in
                                              texts],
                            "Relevance Score": relevance_scores.flatten()
                        }
                        result_df = pd.DataFrame(data).sort_values(by='Relevance Score', ascending=False)
                    else:
                        data = {
                            "URL": valid_urls,
                            "Keywords Found": [{kw: combined_text.count(kw) for kw in keywords} for combined_text in
                                               texts],
                            "Total Matches": [sum(combined_text.count(kw) for kw in keywords) for combined_text in
                                              texts],
                        }
                        result_df = pd.DataFrame(data).sort_values(by='Total Matches', ascending=False)
                    st.write("Scanning complete!")
                    st.dataframe(result_df)

            if invalid_urls:
                st.write("Invalid URLs:")
                inv_urls_df = pd.DataFrame(invalid_urls, columns=["Invalid URL"])
                st.dataframe(inv_urls_df)


if __name__ == "__main__":
    main()
