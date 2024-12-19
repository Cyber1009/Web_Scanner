import scrapy
import re
import random

class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ['https://example.com']

    def __init__(self, urls=None, keywords=None, result_queue=None, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.urls = urls if urls else []
        self.keywords = keywords if keywords else []
        self.result_queue = result_queue  # Store the queue for sending results back

    def start_requests(self):
        """Ensure the spider starts from the URLs passed via Streamlit."""
        if not self.urls:
            self.log("No URLs to scrape.")
            return

        for url in self.urls:
            try:
                yield scrapy.Request(url, callback=self.parse, cb_kwargs={'keywords': self.keywords})
            except Exception as e:
                self.log(f"Error requesting {url}: {e}")

    def parse(self, response, keywords):
        """Main parsing method."""
        page_url = response.url
        page_text = response.text.lower()

        keyword_matches = []
        for keyword in keywords:
            try:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', page_text))  # Match full words
                if count > 0:
                    keyword_matches.append({'Keyword': keyword, 'Count': count})
            except Exception as e:
                self.log(f"Error processing keyword '{keyword}' on {page_url}: {e}")

        if keyword_matches:
            result = {
                'URL': page_url,
                'Keyword Matches': keyword_matches,
                'Total Count': sum(match['Count'] for match in keyword_matches),
            }
        else:
            result = {
                'URL': page_url,
                'Keyword Matches': [],
                'Total Count': 0,  # Explicitly set count to 0 if no matches are found
            }

        if self.result_queue:
            try:
                self.result_queue.put(result)  # Send result back to the main process
                self.log(f"Result added for {page_url}: {result}")
            except Exception as e:
                self.log(f"Error pushing result to queue: {e}")
