import scrapy
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scrapy.crawler import CrawlerProcess



class KeywordScanner(scrapy.Spider):
    name = "keyword_scanner"

    def __init__(self, url_str, keyword_str):
        super(KeywordScanner, self).__init__()
        self.url = url_str
        self.keywords = keyword_str.split()

    def start_requests(self):
        yield scrapy.Request(self.url, callback=self.parse)

    def parse(self, response):
        url = response.url
        text = response.css("body *::text").getall()

        # Combine keywords (phrases and individual words)
        keyword_counts = self.count_keywords(" ".join(text), self.keywords)
        total_count = sum(keyword_counts.values())

        yield {
            "URL": url,
            "Keyword Matches": keyword_counts,

        }

    @classmethod
    def run(cls, url, keywords):
        spider = cls(url, keywords)  # Create an instance of the spider
        process = CrawlerProcess()
        process.crawl(cls, url, keywords)  # Pass the class to crawl method
        process.start()  # Start the crawling process (blocking)

        scan_result = []
        for crawler in process.crawlers:
            for item in crawler.spider.item_passer.items():
                scan_result.append(item)
        return scan_result

    # def start_requests(self):
    #     yield scrapy.Request(self.urls, callback=self.parse)
    #
    # def count_keywords(self, text, keywords):
    #     # Counts the occurrences of each keyword in the text.
    #     keyword_counts = Counter(text.split())
    #     for keyword in keywords:
    #         keyword_counts[keyword.lower()] += keyword_counts.get(keyword, 0)
    #     return keyword_counts
    #
    # def parse(self, response):
    #     url = response.url
    #     text = response.css("body *::text").getall()
    #
    #     # Combine keywords (phrases and individual words)
    #     keyword_counts = self.count_keywords(" ".join(text), self.keywords)
    #     total_count = sum(keyword_counts.values())
    #
    #     # Follow links if enabled
    #     # ... (Implement link following logic if needed)
    #     print(f"URL: {url}")
    #     print(f"Keyword Matches: {keyword_counts}")
    #     print(f"Total Count: {total_count}")
    #
    #     yield {
    #         "URL": url,
    #         "Keyword Matches": keyword_counts,
    #         "Total Count": total_count,
    #     }

    @staticmethod
    def advanced_search(result_df, keywords):
        contents = [content for content in result_df['Keyword Matches'] if content]  # Filter empty content
        stop_words = stopwords.words('english')  # Use English stop words
        tfidf_scores = []
        for content in contents:
            vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, smooth_idf=True)
            tfidf_matrix = vectorizer.fit_transform([content])
            keyword_indices = [vectorizer.vocabulary_[keyword] for keyword in keywords if
                               keyword in vectorizer.vocabulary_]
            keyword_tfidf = tfidf_matrix[0, keyword_indices].mean() if tfidf_matrix[0, keyword_indices].sum() > 0 else 0
            tfidf_scores.append(keyword_tfidf)

        result_df["Relevance"] = tfidf_scores
        result_df = result_df.sort_values(by='Relevance', ascending=False)
        return result_df

# if __name__ == "__main__":
#     from scrapy.crawler import CrawlerProcess
#
#     def main(url, keywords):
#         process = CrawlerProcess()
#         process.crawl(KeywordScanner, url=url, keywords=keywords)
#         process.start()
#
#     if __name__ == "__main__":
#         url = "https://www.example.com"  # Replace with your target URL
#         keywords = ["keyword1", "keyword2"]  # Replace with your keywords
#
#         main(url, keywords)