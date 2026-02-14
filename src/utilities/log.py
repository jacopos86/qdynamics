import logging
import inspect
import sys
import os
from pathlib import Path
from colorlog import ColoredFormatter

#
#   Log class
#

class LogSingleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LogSingleton, cls).__new__(cls)
        return cls._instance

class LogClass(LogSingleton):
    def __init__(self, LOG_LEVEL, logfile):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            format = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
            file_formatter = logging.Formatter(format)
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(file_formatter)
            self.logger  = logging.getLogger()
            self.logger.addHandler(file_handler)
            self.logger.setLevel(LOG_LEVEL)
            self.level = self.logger.getEffectiveLevel()
            self.msg_len_min = 58
    @staticmethod
    def __get_call():
        stack = inspect.stack()
        # stack[1] gives previous function ('info' in our case)
        # stack[2] gives before previous function and so on
        fn = stack[2][1].split('/')
        ln = stack[2][2]
        func = stack[2][3]
        return fn[-1], func, ln
    def _format_message(self, message):
        """Format the message with caller info and log message."""
        fn, func, ln = self.__get_call()
        msg = f"{fn} - {func} at line {ln} : {message:<30}"
        return msg
    def info(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.logger.info(msg2, *args)
    def debug(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.logger.debug(msg2, *args)
    def warning(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.logger.warning(msg2, *args)
    def error(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.logger.error(msg2, *args)
        sys.exit(1)

class ColoredLogClass(LogSingleton):
    def __init__(self, LOG_LEVEL):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            LOG_FORMAT= "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)4s"
            colors={
                'DEBUG':    'blue,bg_white',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'black,bg_red',
                'CRITICAL': 'red,bg_white',
	        }
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOG_FORMAT, log_colors=colors)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)
            self.log = logging.getLogger('pythonConfig')
            self.log.setLevel(LOG_LEVEL)
            self.log.addHandler(stream)
            self.level = self.log.getEffectiveLevel()
            self.msg_len_min = 58
    @staticmethod
    def __get_call():
        stack = inspect.stack()
        # stack[1] gives previous function ('info' in our case)
        # stack[2] gives before previous function and so on
        fn = stack[2][1].split('/')
        ln = stack[2][2]
        func = stack[2][3]
        return fn[-1], func, ln
    def _format_message(self, message):
        """Format the message with caller info and log message."""
        fn, func, ln = self.__get_call()
        msg = f"{fn} - {func} at line {ln} : {message:<30}"
        return msg
    def info(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.log.info(msg2, *args)
    def debug(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.log.debug(msg2, *args)
    def warning(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.log.warning(msg2, *args)
    def error(self, message, *args):
        msg = self._format_message(message).split(':')
        msg2 = msg[0] + ":"
        if len(msg[0]) < self.msg_len_min:
            msg2 += " " * (self.msg_len_min - len(msg[0]))
        msg2 += message
        self.log.error(msg2, *args)
        sys.exit(1)
#
# set up logger
#
def setup_logger():
    LOG_LEVEL = logging.DEBUG   # default
    if "LOG_LEVEL" in os.environ:
        log_level_str = os.environ.get("LOG_LEVEL").upper()
    else:
        log_level_str = "NOTSET"
    if log_level_str in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        LOG_LEVEL = getattr(logging, log_level_str)
    else:
        if "LOG_LEVEL" in os.environ:
            import warnings
            warnings.warn(f"WARNING environment variable \"LOG_LEVEL\" set to {log_level_str} which is not one of the valid options:\n\tDEBUG, INFO, WARNING, ERROR, CRITICAL\nUsing default instead.")
            
        LOG_LEVEL = logging.NOTSET
    COLOR = os.environ.get("COLOR_LOG", "0") == "1"
    # set up logging system
    if COLOR:
        log = ColoredLogClass(LOG_LEVEL)
    else:
        if "LOG_FILE" in os.environ:
            LOGFILE = os.environ.get("LOG_FILE")
        else:
            LOGFILE = "py-dephase.log"
        ## AG should this be the directory of the package or cwd? 
        PACKAGE_DIR = Path.cwd()
        LOGFILE = PACKAGE_DIR / LOGFILE
        log = LogClass(LOG_LEVEL, LOGFILE)
    return log

#
#  set up log object
#

log = setup_logger()
