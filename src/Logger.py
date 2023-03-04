import logging
import os
import datetime


# My Logger is singleton class for adding logging to catch errors and debugging during the
# working of the server

def singleton(cls):
    _instances = {}

    def getinstance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return getinstance


@singleton
class MyLogger:
    _logger = None

    # Executed before Init Function __init__
    def __init__(self):
        print("Logger new")
        self._logger = logging.getLogger("crumbs")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')
        now = datetime.datetime.now()
        dirname = "./log"
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fileHandler = logging.FileHandler(
            dirname + "/log_" + now.strftime("%Y-%m-%d") + ".log")
        streamHandler = logging.StreamHandler()
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

    def getLogger(self):
        return self._logger


"""
# a simple usecase
if __name__ == "__main__":
    logger = MyLogger().getLogger()
    logger.info("Hello, Logger")
    logger = MyLogger().getLogger()
    logger.debug("bug occured")

"""
