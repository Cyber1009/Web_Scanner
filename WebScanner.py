
import httpx
import nltk
import asyncio
import re
import textblob
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from textblob import TextBlob
from textblob import download_corpora

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    textblob.download_corpora()
except Exception as e:
    print("Error downloading corpora for TextBlob:", e)
    raise e  # Reraise to handle gracefully

lemmatizer = WordNetLemmatizer()

def check_url(url_list):
    valid_urls = []
    invalid_urls = []
    for u in url_list:
        uc = u.strip()
        if uc.startswith("www."):
            uc = "https://" + uc
        if not uc.startswith(("http://", "https://")):
            invalid_urls.append(u)
            continue
        parsed = urlparse(uc)
        if parsed.scheme and parsed.netloc:
            valid_urls.append(uc)
        else:
            invalid_urls.append(u)
    return valid_urls, invalid_urls

async def fetch_url(url, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    }

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return url, response.text, True  # Success on HTTPS
        except Exception as e:
            # print(f"Attempt {attempt} failed for {url} with error: {e}")
            if attempt == retries:  # On final failure, try HTTP fallback
                http_url = url.replace("https://", "http://") if url.startswith("https://") else url
                try:
                    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                        response = await client.get(http_url, headers=headers, timeout=10)
                        response.raise_for_status()
                        return http_url, response.text, True  # Success on HTTP fallback
                except Exception as e:
                    # print(f"HTTP fallback failed for {url}: {e}")
                    pass
    return url, None, False  # Final failure after all retries

async def fetch_all_urls(urls):
    progress = st.progress(0)  # Initialize the progress bar
    total_urls = len(urls)
    batch_size = 30  # total_urls/10
    results = []

    for i in range(0, total_urls, batch_size):
        batch = urls[i:i + batch_size]
        tasks = [fetch_url(url) for url in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)

        # Update progress bar incrementally
        progress.progress(min((i + batch_size) / total_urls, 1.0))

    progress.empty()  # Clear progress bar after completion
    return results

def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    headers = " ".join([h.get_text(separator=" ", strip=True) for h in soup.find_all(["h1", "h2"])])
    body = soup.body.get_text(separator=" ", strip=True) if soup.body else ""
    text = headers + " " + body  # Combine headers with body text
    return re.sub(r"\s+", " ", text.lower())  # Clean up extra spaces and normalize to lowercase

def process_text(text):
    words = TextBlob(text).words
    return ' '.join([word.lemmatize() for word in words])

def analyze_texts(texts, keywords, flexible_search=False, advanced_search=False):

    # Initialize results
    keyword_matches = []
    total_matches = []
    relevance_scores = None

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
        beta = 0.3   # Weight for total matches
        hybrid_scores = [
            alpha * similarity + beta * (matches / (max(total_matches)+1))  # Normalize matches
            for similarity, matches in zip(cosine_similarities, total_matches)
        ]

        # Normalize the hybrid scores to the range [0, 1]
        min_score, max_score = min(hybrid_scores), max(hybrid_scores)
        if max_score > min_score:
            relevance_scores = [(score - min_score) / (max_score - min_score) for score in hybrid_scores]

        # # Calculate the relevance scores
        # relevance_scores = list(cosine_similarities)
        #
        # # Normalize the relevance scores to the range [0, 1]
        # min_score, max_score = min(relevance_scores), max(relevance_scores)
        # if max_score > min_score:
        #     relevance_scores = [(score - min_score) / (max_score - min_score) for score in relevance_scores]

    return keyword_matches, total_matches, relevance_scores

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
                                      help="This option will match keywords with their base form, ignoring plural forms and verb tenses, to increase the flexibility of the search.")
        advanced_search = st.checkbox("Enable Advanced Search",
                                      help="This option calculates the relevance score of each website using advanced algorithms. Scanning might take longer.")

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

                # Initialize lists to hold URL and text content
                texts, valid_urls = [], []

                # Fetch and parse HTML for each valid URL
                results = asyncio.run(fetch_all_urls(urls))

                for url, content, success in results:
                    if success:
                        text = parse_html(content)
                        texts.append(text)
                        valid_urls.append(url)
                    else:
                        invalid_urls.append(url)

                # Calculate relevance scores (including flexible search if enabled)
                if texts:
                    keyword_matches, total_matches, relevance_scores = analyze_texts(
                        texts=texts,
                        keywords=keywords,
                        flexible_search=flexible_search,
                        advanced_search=advanced_search
                    )

                    data = {
                        "URL": valid_urls,
                        "Keywords Found": keyword_matches,
                        "Total Matches": total_matches,
                    }

                    result_df = pd.DataFrame(data)

                    if advanced_search:
                        result_df["Relevance Score"] = relevance_scores
                        result_df = result_df.sort_values(by='Relevance Score', ascending=False)
                    else:
                        result_df = result_df.sort_values(by='Total Matches', ascending=False)

                    st.write("Scanning complete!")
                    st.dataframe(result_df, column_config={
                        "URL": st.column_config.LinkColumn(width="medium")
                    })

            if invalid_urls:
                st.write("Invalid URLs:")
                inv_urls_df = pd.DataFrame(invalid_urls, columns=["Invalid URL"])
                st.dataframe(inv_urls_df, column_config={
                    "Invalid URL": st.column_config.LinkColumn()
                })


if __name__ == "__main__":
    main()
