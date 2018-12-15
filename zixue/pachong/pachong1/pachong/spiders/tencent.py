# -*- coding: utf-8 -*-
import scrapy


class DianyingSpider(scrapy.Spider):
    name = 'tencent'
    allowed_domains = ['tencent.com']
    start_urls = ['https://hr.tencent.com/position.php']

    def parse(self, response):
        tr_list = response.xpath("//table[@class='tablelist']/tr")[1:-1]

        for tr in tr_list:
            item = {}
            item['title'] = tr.xpath("./td[1]/a/text()").extract_first()
            item['position'] = tr.xpath("./td[2]/a/text()").extract_first()
            item['publish_date'] = tr.xpath("./td[5]/a/text()").extract_first()
            #print(item)
            yield item
