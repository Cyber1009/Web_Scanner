import scrapy
import re
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider

class MySpider(scrapy.Spider):
    name = 'my_spider'

    def __init__(self, urls=None, keywords=None, crawl_subpages=True, max_subpages=5, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.urls = urls if urls else []
        self.keywords = keywords if keywords else []
        self.crawl_subpages = crawl_subpages  # Option to crawl subpages
        self.max_subpages = max_subpages  # Limit on the number of subpages to crawl

        # Ensure URLs are properly formatted
        self.urls = [url.strip() for url in self.urls if url.strip()]
        self.urls = [url if url.startswith(('http://', 'https://')) else 'https://' + url for url in self.urls]

        # Handle allowed domains gracefully
        self.allowed_domains = []
        for url in self.urls:
            try:
                domain = url.split('//')[1].split('/')[0]
                self.allowed_domains.append(domain)
            except IndexError:
                self.log(f"Skipping invalid URL: {url}")

        self.allowed_domains = list(set(self.allowed_domains))  # Remove duplicates

        # Allow only the specified URLs
        self.start_urls = self.urls

    def start_requests(self):
        """Ensure the spider starts from the URLs passed via Streamlit."""
        if not self.urls:
            self.log("No URLs to scrape.")
            return
        for url in self.urls:
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={'keywords': self.keywords})

    def parse(self, response, keywords):
        """Main parsing method."""
        page_url = response.url
        page_text = response.text.lower()  # Ensure the text is lowercased for case-insensitive matching

        # Extract readable text (remove HTML tags) using Scrapy's response.text
        page_text = response.css('body').xpath('string()').get().lower()

        keyword_matches = []
        for keyword in keywords:
            count = len(
                re.findall(r'\b' + re.escape(keyword) + r'\b', page_text))  # Use word boundaries for exact match
            if count > 0:
                keyword_matches.append((keyword, count))

        if keyword_matches:
            result = {
                'URL': page_url,
                'Keyword Matches': ', '.join([f"{kw}: {count}" for kw, count in keyword_matches]),
                'Total Count': sum(count for _, count in keyword_matches),
            }
            yield result  # Return the result to Scrapy's pipeline (directly back to Streamlit)

        # Handle subpage crawling if enabled
        if self.crawl_subpages:
            self.crawl_subpages_func(response)

    def crawl_subpages_func(self, response):
        """Handle subpage crawling."""
        links = response.css('a::attr(href)').getall()
        links = [response.urljoin(link) for link in links]

        # Filter links to only those that are in the allowed domains
        links = [link for link in links if any(domain in link for domain in self.allowed_domains)]

        # Limit the number of subpages to crawl
        links = links[:self.max_subpages]

        for link in links:
            yield scrapy.Request(link, callback=self.parse, cb_kwargs={'keywords': self.keywords})

