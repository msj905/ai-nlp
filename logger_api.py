# coding=utf-8
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s [%(fliename)]s:%(lineon)]'
                    ': %(message)'
                    ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]'
                    )


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.info("this is a")


