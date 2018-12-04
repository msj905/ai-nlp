# -*- coding: utf-8 -*-
import scrapy


class DinayingSpider(scrapy.Spider):
    name = 'dinaying'
    allowed_domains = ['www.lalalo.com']
    start_urls = ['http://www.lalalo.com/type/1.html']

    def parse(self, response):
	print('=='*40)
        print(respone)
	print('=='*40)


