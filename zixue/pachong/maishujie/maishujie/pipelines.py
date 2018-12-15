# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import logging

logger = logging.getLogger(__name__)

class MaishujiePipeline(object):
    def process_item(self, item, spider):
        #TODD
        print(item)
        return item


class MaishujiePipeline1(object):
    def process_item(self, item, spider):
        #TODD
        item["hello"] = "world"
        return item