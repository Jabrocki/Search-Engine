import scrapy
from ..items import BookItem, ChunkItem
import urllib.parse as urlparse
from config import NO_BOOKS, NO_WORDS_PER_CHUNK, OVERLAP_SIZE, MAX_NO_OF_CHUNKS


class GutenbergOrgSpider(scrapy.Spider):
    name = "gutenberg_org"
    allowed_domains = ["gutenberg.org"]

    doc_count = 0

    def start_requests(self):
        for i in range(1, NO_BOOKS + 1):
            print(f"Scheduling book ID: {i}")
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
            # Skip dictionaries, encyclopedias, and CIA documents
            if (
                "dictionary" in title.lower()
                or "encyclopedia" in title.lower()
                or "cia" in title.lower()
            ):
                return

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
        if self.doc_count >= MAX_NO_OF_CHUNKS:
            return
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

        for i in range(0, len(all_words), NO_WORDS_PER_CHUNK - OVERLAP_SIZE):
            if self.doc_count >= MAX_NO_OF_CHUNKS:
                return
            word_slice = all_words[i : i + NO_WORDS_PER_CHUNK]
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
