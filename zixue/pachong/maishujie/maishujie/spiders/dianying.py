# -*- coding: utf-8 -*-
import scrapy


class DianyingSpider(scrapy.Spider):
    name = 'dianying'
    allowed_domains = ['www.lalalo.com']
    start_urls = ['http://www.lalalo.com/']

    def parse(self, response):
        pass
