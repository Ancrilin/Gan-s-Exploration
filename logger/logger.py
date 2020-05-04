# coding:utf-8

import logging

class Logger:
    def __init__(self, log_save_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # output log to file
        handler = logging.FileHandler(log_save_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # output log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        # set them to logging attributes
        logger.addHandler(handler)
        logger.addHandler(console)

        #
        logger.removeHandler("stderr")

        # set self attribute
        self.log = logger

    def info(self, message):
        self.log.info(message)

    def debug(self, message):
        self.log.debug(message)

    def warning(self, message):
        self.log.warning(message)
