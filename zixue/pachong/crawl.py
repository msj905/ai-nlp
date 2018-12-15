# -*- coding: utf-8 -*-
import scrapy


class CrawlSpider(scrapy.Spider):
    name = 'crawl'
    allowed_domains = ['tt']
    start_urls = ['http://tt/']

    def parse(self, response):
        pass
