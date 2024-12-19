from fastapi import FastAPI, File, UploadFile, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import pandas as pd
import asyncio
from playwright.async_api import async_playwright

app = FastAPI()


@app.post("/scan/")
async def scan_urls(file: UploadFile, keywords: str = Form(...)):
    # Step 1: Load URLs from the uploaded file
    df = pd.read_csv(file.file)
    urls = df['url'].tolist()
    search_terms = keywords.split(",")  # Split keywords into a list

    # Step 2: Scrape the URLs
    async def fetch_content(url):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=15000)
                content = await page.content()
            except Exception:
                content = ""
            await browser.close()
            return content

    tasks = [fetch_content(url) for url in urls]
    scraped_content = await asyncio.gather(*tasks)

    # Step 3: Calculate TF-IDF relevance scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(scraped_content)
    keyword_scores = [
        vectorizer.transform([kw]).mean(axis=1).max() for kw in search_terms
    ]

    # Step 4: Score and rank sites
    scores = [
        sum([fuzz.partial_ratio(content, kw) for kw in search_terms]) for content in scraped_content
    ]
    results = pd.DataFrame({
        "url": urls,
        "content": scraped_content,
        "tfidf_score": scores
    }).sort_values(by="tfidf_score", ascending=False)

    # Step 5: Return Results
    return {
        "top_results": results[["url", "tfidf_score"]].head(5).to_dict(orient="records")
    }
