import os
import sys
import json
import datetime
import logging


"""
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""

def get_timestamp():
    return datetime.datetime.now().strftime('_%y%m%d_%H%M%S')

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


"""
# --------------------------------------------
# logger
# --------------------------------------------
"""


def logger_info(logger_name, log_path="default_logger.log"):
    """set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    """
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print("LogHandlers exist!")
    else:
        print("LogHandlers setup!")
        level = logging.INFO
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d : %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


"""
# --------------------------------------------
# print to file and std_out simultaneously
# --------------------------------------------
"""


class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass


def save(opt, save_path):
    opt_path = opt['option_path']
    opt_path_copy = save_path
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    if opt["train"]["resume"]["finetune"]:
        filename = 'finetune'
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)