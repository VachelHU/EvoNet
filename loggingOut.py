# -*- coding: utf-8 -*-

import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='run.log',
                filemode='a')

#################################################################################################
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
#################################################################################################

def info(str):
    return logging.info(str)

def warning(str):
    return logging.warning(str)

def error(str):
    return logging.error(str)