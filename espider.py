import scrapy
import re


class MySpider(scrapy.Spider):
    name = "my_spider"

    def __init__(self, urls=None, keywords=None, invalid_urls=None, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.urls = urls if urls else []
        self.keywords = keywords if keywords else []
        self.invalid_urls = invalid_urls if invalid_urls else []

    def start_requests(self):
        """Generate requests for the provided URLs."""
        if not self.urls:
            self.log("No URLs to scrape.")
            return

        for url in self.urls:
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={'keywords': self.keywords}, errback=self.handle_error)

    def parse(self, response, keywords):
        """Parse the response and look for keyword matches."""
        page_url = response.url
        page_text = response.text.lower()

        keyword_matches = []
        for keyword in keywords:
            try:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', page_text))
                if count > 0:
                    keyword_matches.append({'Keyword': keyword, 'Count': count})
            except Exception as e:
                self.log(f"Error processing keyword '{keyword}' on {page_url}: {e}")

        result = {
            'URL': page_url,
            'Keyword Matches': keyword_matches,
            'Total Count': sum(match['Count'] for match in keyword_matches),
        }

        yield result

    def handle_error(self, failure):
        """Handle errors for failed requests."""
        url = failure.request.url
        self.invalid_urls.append(url)
        self.log(f"Error with {url}: {failure.value}")

    def closed(self, reason):
        """Called when the spider is closed."""
        self.log(f"Spider closed: {reason}")
