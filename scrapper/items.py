# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BookItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    author = scrapy.Field()
    image_url = scrapy.Field()
    book_url = scrapy.Field()


class ChunkItem(scrapy.Item):
    book_id = scrapy.Field()
    chunk_url = scrapy.Field()
    text = scrapy.Field()
