import scrapy
import re
import random

class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ['https://example.com']

    # List of common User-Agent strings
    USER_AGENT_LIST = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'
    ]

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
            # Pass keywords to parse via 'cb_kwargs'
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={'keywords': self.keywords})

    def parse(self, response, keywords):
        """Main parsing method."""
        page_url = response.url
        page_text = response.text.lower()

        keyword_matches = []
        for keyword in keywords:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', page_text))  # Match full words
            if count > 0:
                keyword_matches.append({'Keyword': keyword, 'Count': count})

        if keyword_matches:
            result = {
                'URL': page_url,
                'Keyword Matches': keyword_matches,
                'Total Count': sum(match['Count'] for match in keyword_matches),
            }
            if self.result_queue:
                try:
                    self.result_queue.put(result)  # Send result back to the main process
                    self.log(f"Result added for {page_url}: {result}")
                except Exception as e:
                    self.log(f"Error pushing result to queue: {e}")
        else:
            self.log(f"No keywords found on {page_url}")


