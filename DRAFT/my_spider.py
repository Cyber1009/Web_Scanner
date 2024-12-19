import scrapy
import re

class MySpider(scrapy.Spider):
    name = 'my_spider'

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
            count = len(re.findall(keyword, page_text))  # Count keyword matches
            if count > 0:
                keyword_matches.append((keyword, count))

        if keyword_matches:
            result = {
                'URL': page_url,
                'Keyword Matches': ', '.join([f"{kw}: {count}" for kw, count in keyword_matches]),
                'Total Count': sum(count for _, count in keyword_matches),
            }
            if self.result_queue:
                self.result_queue.put(result)  # Send result back to the main process
        else:
            self.log(f"No keywords found on {page_url}")
