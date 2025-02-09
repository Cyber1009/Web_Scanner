from os import write

import httpx
import nltk
import asyncio
import re
import pandas as pd
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urljoin, urlparse


# from streamlit import session_state

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# try:
#     nltk.download('punkt_tab')
# except:
#     pass
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


lemmatizer = WordNetLemmatizer()

def check_url(url_list):
    valid_urls = []
    invalid_urls = []

    for u in url_list:
        uc = u.strip()

        # Automatically prepend "https://" only if it starts with "www."
        if uc.startswith("www."):
            uc = "https://" + uc

        # Parse the URL
        parsed = urlparse(uc)

        # Check if the URL is valid
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            valid_urls.append(uc)
        else:
            invalid_urls.append((u, "Invalid Format"))

    return valid_urls, invalid_urls


async def fetch_url(client, url, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }
    error_message = "Unknown Error"
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return url, response.text, True
        except Exception as e:
            error_message = str(e)
    return url, error_message, False

async def fetch_all_urls(urls):
    retries = 2
    progress = st.progress(0)
    # progress_text = st.empty()
    total_urls = len(urls)
    batch_size = min(30, total_urls)
    results = []
    failed_urls = []
    invalid_urls = []
    # Phase 1: Use a client with verify=True
    async with httpx.AsyncClient(follow_redirects=True, verify=True) as client:
        for i in range(0, total_urls, batch_size):
            batch = urls[i:i + batch_size]
            tasks = [fetch_url(client, url, retries=retries) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in batch_results:
                if isinstance(res, Exception):
                    # In rare cases, an exception may be raised by gather.
                    continue
                url_fetched, content, success = res
                if success:
                    results.append((url_fetched, content))
                else:
                    modified_url = url_fetched.replace("https://", "http://") if url_fetched.startswith("https://") else url_fetched
                    failed_urls.append(modified_url)
            progress.progress(min((i + batch_size) / total_urls, 1.0))
        # progress_text.write(f"(Progress for current url: {i}/{total_urls})")


    # Phase 2: Retry failed URLs using a client with verify=False
    # failed_urls = [url for url, (u, content, success) in results_dict.items() if not success]
    if failed_urls:
        async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
            tasks = [fetch_url(client, url, retries=retries) for url in failed_urls]
            retry_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in retry_results:
                if isinstance(res, Exception):
                    continue
                url_fetched, content, success = res
                if success:
                    results.append((url_fetched, content))
                else:
                    invalid_urls.append((url_fetched,content))
            #     results_dict[url_fetched] = (url_fetched, content, success)
            #     return url, error_message, False
    progress.empty()
    return results, invalid_urls

def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    # soup = BeautifulSoup(content, "lxml")

    # Remove unwanted tags such as <script>, <style>, and comments
    for element in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        element.decompose()

    # Extract all visible text from the body
    body = soup.get_text(separator=" ", strip=True)

    # Clean up any leftover whitespace and normalize to lowercase
    readable_text = re.sub(r"\s+", " ", body)

    return readable_text

def parse_html_sub(base_url, content):
    soup = BeautifulSoup(content, "html.parser")
    # Extract all links
    subpage_urls = set()  # Use a set to avoid duplicates
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Convert relative URLs to absolute URLs
        full_url = urljoin(base_url, href)
        # Validate the URL
        if is_valid_url(full_url, base_url):
            subpage_urls.add(full_url)
    return list(subpage_urls)

def process_text(text):
    # Tokenize the text using NLTK's word_tokenize
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def analyze_texts(texts, keywords, flexible_search=False, advanced_search=False):
    # Initialize results
    keyword_matches = []
    total_matches = []
    relevance_scores = None

    if not texts or not keywords:
        st.error("No valid text or keywords available for analysis.")
        return [], [], None

    # Preprocess keywords and texts if flexible search is enabled
    if flexible_search:
        keyword_patterns = {}
        processed_keywords = []
        for kw in keywords:
            p_kw = process_text(kw)
            processed_keywords.append(p_kw)
            keyword_patterns[kw] = re.compile(rf"\b{re.escape(p_kw)}\b", re.IGNORECASE)

        keywords = processed_keywords
        texts = [process_text(text) for text in texts]
    else:
        keyword_patterns = {
            kw: re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords
        }

    for i, text in enumerate(texts):
        # Calculate keyword matches
        matches = {kw: len(re.findall(pattern, text)) for kw, pattern in keyword_patterns.items()}
        total_match_count = sum(matches.values())
        keyword_matches.append(matches)
        total_matches.append(total_match_count)

    # Perform TF-IDF analysis if advanced search is enabled
    # Perform Cosine Similarity analysis if advanced search is enabled
    if advanced_search:
        # Combine keywords into a single string
        keyword_phrase = " ".join(keywords)

        # Create the content list, with the keyword phrase as the first item
        content = [keyword_phrase] + texts  # Add keywords as the first item (the query)

        # Create the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, len(keyword_phrase.split())))
        # Vectorize the content (keyword phrase + websites)
        tfidf_matrix = vectorizer.fit_transform(content)

        # Calculate cosine similarity between the keyword phrase and each website's content
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Combine cosine similarity with total keyword matches
        alpha = 0.7  # Weight for cosine similarity
        beta = 0.3  # Weight for total matches
        hybrid_scores = [
            alpha * similarity + beta * (matches / (max(total_matches) + 1))  # Normalize matches
            for similarity, matches in zip(cosine_similarities, total_matches)
        ]

        # Normalize the hybrid scores to the range [0, 1]
        min_score, max_score = min(hybrid_scores), max(hybrid_scores)
        if max_score > min_score:
            relevance_scores = [(score - min_score) / (max_score - min_score) for score in hybrid_scores]

    return keyword_matches, total_matches, relevance_scores

def is_valid_url(url, base_url):
    """Validate if the URL belongs to the same domain as the base URL."""
    parsed_base = urlparse(base_url)
    parsed_url = urlparse(url)

    # Check if it's the same domain and a valid HTTP/HTTPS URL
    return parsed_url.scheme in ['http', 'https'] and parsed_base.netloc == parsed_url.netloc

def main():

    st.title("Website Keyword Scanner")
    st.write(" ")
    inv_urls = []
    # inac_urls = []

    if "valid_urls" not in st.session_state:
        st.session_state.valid_urls = []
    # if "invalid_urls" not in st.session_state:
    #     st.session_state.invalid_urls = []
    if "option" not in st.session_state:
        st.session_state.option = []
    # if "keywords" not in st.session_state:
    #     st.session_state.keywords = []
    if 'data' not in st.session_state:
        st.session_state.key = 'data'
        st.session_state.data = None

    tab1, tab2, tab3 = st.tabs(["Enter URLs manually", "Upload file with URLs", 'Search directly from result file'])

    with tab1:

    # with st.expander('Enter or upload URLs:', expanded=True):

        st.subheader("Enter URLs for scanning:")
        col1, col2, col3 = st.columns([6, 1, 1], vertical_alignment="bottom")

        with col1:
            e_urls_input = st.text_input(" ", placeholder="Enter URLs here (Comma-separated)")

        with col2:
            e_button = st.button("Enter")

        # with col3:
        #     s_button = st.button("Sub-link")

        if e_button:

            raw_e_urls = list(set([u.strip().lower() for u in e_urls_input.split(',') if u.strip()]))
            try:
                e_urls, e_invalid_urls = check_url(raw_e_urls)
                if not e_urls:
                    st.error("Please enter valid URLs.")
                    return
                else:
                    st.session_state.valid_urls = e_urls
                    inv_urls = e_invalid_urls
                    # st.session_state.invalid_urls = e_invalid_urls
                    # st.session_state.option = "User input"
            except Exception:
                st.error("Failed to read the URL entered. Please try again.")
                return

    with tab2:
        st.subheader("Upload a file with URLs:")
        uploaded_file = st.file_uploader(" ", type=["csv", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.dataframe(df.head())

            # Select column
            sc_t_col, sc_u_col, sc_b_col = st.columns([3,2,2], vertical_alignment="bottom")
            with sc_t_col:
                st.write("Select the column that contains valid URLs:")

            with sc_u_col:
                url_column = st.selectbox(" ",
                                          df.columns,
                                          index=None,
                                          placeholder="select url column...",
                                          )
            with sc_b_col:
                b_button = st.button("Read URLs from column")

            if url_column:
                try:
                    c_urls = df[url_column].dropna().tolist()
                    f_urls, f_invalid_urls = check_url(c_urls)
                    if not f_urls:
                        st.error(f"No valid URLs found in column '{url_column}'.")
                        return
                    elif b_button:
                        st.session_state.valid_urls = f_urls
                        # st.session_state.invalid_urls = f_invalid_urls
                        inv_urls = f_invalid_urls
                        st.session_state.option = uploaded_file.name + " , column '" + url_column + "'"
                except Exception:
                    st.error(f"Cannot read the URL column '{url_column}'.")
                    return

    with tab3:
        st.subheader("Upload a result file generated by the web scanner:")
        uploaded_file = st.file_uploader("   ", type=["csv"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                up_df = pd.read_csv(uploaded_file, index_col=0)
            else:
                st.error('invalid file format')
            st.dataframe(up_df.head())
            if st.button('Search from this file'):
                if 'Content' in up_df.columns:
                    st.session_state.data = up_df
                else:
                    st.error('Invalid file format')


    if st.session_state.valid_urls and st.session_state.option:

        urls = st.session_state.valid_urls
        # inv_urls = inv_urls
        # st.divider()
        st.write(" ")
        with st.container(border=True):
            st.markdown(f" {len(urls)} url(s) detected from ***{st.session_state.option}*** :" )
            st.write(urls[:5] + (["..."] if len(urls) > 5 else []))
            # st.write(f" {len(invalid_urls)} invalid urls found")

        col4, col5, col6 = st.columns([1, 5, 1], vertical_alignment="bottom")
        with col4:
            crawl_button = st.button("Crawl")
        with col5:
            sub_option = st.checkbox("Crawl all Subpages",
                                          help="This option will fetch content from all subpages of all given links. Significantly increase scanning time.")

        if crawl_button:
            # invalid_urls = inv_url
            with st.spinner("Scanning websites... This may take a while."):
                # valid_urls = []
                urls_data = []
                results = asyncio.run(fetch_all_urls(urls))
                urls_cont, inac_urls = results
                inv_urls.extend(inac_urls)
                # st.write('ewew')
                # st.write(inac_urls)
                # st.write('sfa')
                # st.write(inv_urls)
                for url, content in urls_cont:
                    # if success:
                    url_data = {
                        "Main URL": url,
                        "Subpage URL": url,
                        "Content": parse_html(content)
                        # "Keywords Found": "keyword1, keyword2",
                    }
                    urls_data.append(url_data)

                    sub_links = parse_html_sub(url, content) if sub_option else False

                    if sub_option and sub_links:
                        sub_results = asyncio.run(fetch_all_urls(sub_links))
                        sub_urls_cont, sub_inac_urls = sub_results
                        # inv_urls.extend(sub_inac_urls)
                        for s_url, content in sub_urls_cont:
                            sub_data = {
                                "Main URL": url,
                                "Subpage URL": s_url,
                                "Content": parse_html(content)
                                # "Keywords Found": "keyword1, keyword2",
                            }
                            urls_data.append(sub_data)

                    # valid_urls.append(url)

                # inv_urls.extend(inac_urls)
            c_data = pd.DataFrame(urls_data)

            st.session_state.data = c_data
            # data_exist = True
            if sub_option and sub_links:
                st.dataframe(c_data, column_config={
                    "URL": st.column_config.LinkColumn(width="medium")
                })
            else:
                st.dataframe(c_data[["Main URL","Content"]], column_config={
                        "URL": st.column_config.LinkColumn(width="medium")
                    })
            # if inv_urls:
            #     inv_urls_df = pd.DataFrame(inv_urls, columns=["Invalid URL"])
            # if inac_urls:
            #     inac_urls_df = pd.DataFrame(inac_urls, columns=["Invalid URL"])
            #
            #
            if inv_urls:
                st.write("Invalid URLs:")
                # st.write(inv_urls)
                # inv_urls_df = pd.DataFrame(inv_urls)
                inv_urls_df = pd.DataFrame(inv_urls, columns=["Invalid URL", "Status"])
                # # inv_urls_df["Status"] = "invalid"
                st.dataframe(inv_urls_df, column_config={
                    "Status": st.column_config.LinkColumn(width="medium")
                })
                # st.dataframe(inv_urls_df, column_config={
                #     "Status": st.column_config.LinkColumn(width="medium")
                # })

        # if st.session_state.data:
        # if c_data_exist = True
    if st.session_state.data is not None:
        col_keys, col_ad = st.columns([9,2], vertical_alignment="bottom")
        # # keywords_input = st.text_input("Enter keywords/phrases (Comma-separated):")
        with col_keys:
            keywords_input = st.text_input("Keywords:", placeholder="Enter keywords/phrases (Comma-separated):")
        with col_ad:
            with st.popover("Advanced"):
            #     # st.write('Detailed information hidden here')
            # flexible_search = st.checkbox("Enable Flexible Search")
                flexible_search = st.checkbox("Enable Flexible Search",
                                              help="This option will match keywords and website contents with their base form, ignoring plural forms and verb tenses, to increase the flexibility of the search.")

                advanced_search = st.checkbox("Enable Advanced Search",
                                              help="This option calculates the relevance score (0 to 1) of each website using advanced algorithms. The website that is the most relevant to the keywords is given a score of 1.")

        search_button = st.button("Search")
        if search_button:
            keywords = list(set(kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()))
            if not keywords:
                st.error("Please enter valid keywords or phrases")
                return

            with st.spinner("Searching contents... This may take a while."):

                keyword_matches, total_matches, relevance_scores = analyze_texts(
                    texts= list(st.session_state.data["Content"]),
                    keywords=keywords,
                    flexible_search=flexible_search,
                    advanced_search=advanced_search
                )

                st.session_state.data["Keywords Found"] = keyword_matches
                st.session_state.data["Total Matches"] =  total_matches

                result_df = pd.DataFrame(st.session_state.data)

                if advanced_search:
                    result_df["Relevance Score"] = relevance_scores
                    result_df = result_df.sort_values(by='Relevance Score', ascending=False)
                else:
                    result_df = result_df.sort_values(by='Total Matches', ascending=False)

                st.write("***Searching complete!***")
                st.dataframe(result_df, column_config={
                    "URL": st.column_config.LinkColumn(width="medium")
                })


if __name__ == "__main__":
    main()
