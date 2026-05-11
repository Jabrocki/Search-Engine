import scrapy
from ..items import BookItem, ChunkItem
import urllib.parse as urlparse


class GutenbergOrgSpider(scrapy.Spider):
    name = "gutenberg_org"
    allowed_domains = ["gutenberg.org"]

    # Licznik dokumentów (chunków)
    doc_count = 0

    custom_settings = {
        "CONCURRENT_REQUESTS": 16,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 16,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 12.0,
        "DOWNLOAD_DELAY": 0.25,
        "USER_AGENT": "Jan Jabrocki AGH Student, sorry if too many requests. Doing it for school project. Contact: jjabrocki@gmail.com",
        "RETRY_TIMES": 5,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 408],
        "COOKIES_ENABLED": False,
    }

    def start_requests(self):
        for i in range(1, 6000):
            yield scrapy.Request(
                f"https://www.gutenberg.org/ebooks/{i}", callback=self.parse_info
            )

    def parse_info(self, response):
        lang = response.css('tr[itemprop="inLanguage"]::attr(content)').get()

        if lang == "en":
            book_id = response.url.split("/")[-1]
            book_url = response.css('a[title="Read Now!"]::attr(href)').get()
            image_url = response.css("img.cover-art::attr(src)").get()
            author = response.css('[itemprop="creator"]::text').get()
            title_parts = response.css('[itemprop="headline"]::text').getall()
            title = " ".join([t.strip() for t in title_parts if t.strip()])

            item = BookItem(
                id=book_id,
                title=title,
                author=author,
                image_url=image_url,
                book_url=book_url,
            )

            yield item

            if book_url:
                yield scrapy.Request(
                    response.urljoin(book_url),
                    callback=self.parse_book,
                    meta={"book_id": book_id},
                )

    def parse_book(self, response):
        book_id = response.meta["book_id"]

        sel = response.selector

        for junk in sel.xpath(
            '//*[@class="pg-boilerplate"] | //*[@id="pg-header"] | //*[@id="pg-footer"]'
        ):
            junk.root.getparent().remove(junk.root)

        elements = sel.xpath(".//p | .//h1 | .//h2 | .//h3 | .//h4 | .//h5")

        text_blocks = []
        for el in elements:
            clean_text = el.xpath("normalize-space(.)").get()
            if clean_text:
                text_blocks.append(clean_text)

        all_words = " ".join(text_blocks).split()

        chunk_size = 1200
        overlap = 80

        for i in range(0, len(all_words), chunk_size - overlap):
            word_slice = all_words[i : i + chunk_size]
            if len(word_slice) < 50:
                continue

            chunk_text = " ".join(word_slice)

            anchor_text = self.get_safe_anchor(word_slice)

            if anchor_text:
                encoded_anchor = urlparse.quote(anchor_text)
                chunk_url = f"{response.url}#:~:text={encoded_anchor}"
            else:
                chunk_url = response.url

            self.doc_count += 1
            yield ChunkItem(book_id=book_id, chunk_url=chunk_url, text=chunk_text)

    def get_safe_anchor(self, words):
        if len(words) >= 10:
            return " ".join(words[1:10])
        return " ".join(words)
