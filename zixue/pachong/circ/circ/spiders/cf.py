# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import re


#scrapy genspider -t crawl cf circ.gov.cn


class CfSpider(CrawlSpider):
    name = 'cf'
    allowed_domains = ['circ.gov.cn']
    start_urls = ['http://circ.gov.cn/web/site0/tab5240/module14430/page1.htm']
    #定义url规则
    rules = (
        Rule(LinkExtractor(allow=r'/web/site0/tab5240/info\d+\.htm'), callback='parse_item'),
        Rule(LinkExtractor(allow=r'/web/site0/tab5240/module14430/page\d+\.htm'), follow = True),
        # LinkExtractor( 提取url
        # callback = 提取url地质的response交给callback处理
        # follow = True)是否通过rules提取

    )
#pqrse不能定义
    def parse_item(self, response):
        #i = {}
        #i['domain_id'] = response.xpath('//input[@id="sid"]/@value').extract()
        #i['name'] = response.xpath('//div[@id="name"]').extract()
        #i['description'] = response.xpath('//div[@id="description"]').extract()
        #return i
        item = {}
        item["title"] = re.findall("<!--TitleStart-->(.*?)<!--TitleEnd-->",response.body.decode())[0]
        item["publish_date"] = re.findall("发布时间: (20\d{2}-\d{2}-\d{2})",response.body.decode())[0]

        print(item)
