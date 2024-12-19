import scrapy
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scrapy.crawler import CrawlerProcess


class KeywordScanner(scrapy.Spider):
    name = "keyword_scanner"

    def __init__(self, url, keyword_str):
        super(KeywordScanner, self).__init__()
        self.url = url
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
            "Total Count": total_count,
        }

    @classmethod
    def run(cls, url, keywords):
        spider = cls(url, keywords)  # Create an instance of the spider with a single URL
        process = CrawlerProcess()
        process.crawl(cls, url, keywords)  # Pass the class, URL, and keywords
        process.start()  # Start the crawling process (blocking)

        scan_result = []
        for crawler in process.crawlers:
            for item in crawler.spider.item_passer.items():
                scan_result.append(item)
        return scan_result