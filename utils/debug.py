"""
jvlee_LIBS_ML > utils > debug.py

Function definitions for creating and logging to a log files. Used in most other
scripts as the primary method of printing so that things aren't only printed in 
the terminal, but also in an associated log file.

"""

print('debug.py loading...')

# region Imports
# region plain
import sys, os 
import logging
import time
# endregion

# region from
from logging import handlers
from datetime import datetime 
from pathlib import Path
from typing import Optional, IO
# endregion
# endregion

def log(logger,msg):
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)


def get_worker_logger(name):
    return logging.getLogger(f'worker.{name}')


def safe_open_log(filepath, max_retries=3, delay=0.5):
    """Try to open log file with retries for locked files."""
    Path(filepath).parent.parent.mkdir(parents=True,exist_ok=True)
    Path(filepath).parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(max_retries):
        try:
            return open(filepath, 'a', encoding='utf-8')
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"Log file locked, retrying in {delay}s...", file=sys.__stderr__)
                time.sleep(delay)
            else:
                raise


class Logger(logging.Handler):
    """
    A logging.Handler that mirrors records to botht the terminal and a log file.
    Can be used as:
        - A direct Handler: logging.getLogger().addHandler(Logger(path))
        - A QueueListener destination for multiprocessing-safe logging
        - A sys.stdout drop-in(write/flush methods still present)
    """
    def __init__(self, filepath):
        super().__init__()
        self.terminal = sys.__stdout__
        self.log: Optional[IO[str]] = None
        try:
            self.log = safe_open_log(filepath=filepath)
            if self.log is None:
                raise RuntimeError(f"Failed to open log file: {filepath}")
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            separator = f'\n{"="*40}\n{timestamp}\n{"="*50}\n'
            self.log.write(separator)
            self.log.flush()
        except Exception as e:
            if self.terminal is not None:
                self.terminal.write(f"Failed to open log file: {e}\n")
            raise

    def emit(self, record):
        msg = self.format(record) + '\n'
        if self.terminal is not None:
            self.terminal.write(msg)
        if self.log is not None:
            self.log.write(msg)
            self.log.flush()

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        if self.log is not None:
            try:
                self.log.flush()
            except (ValueError, AttributeError):
                pass  # File already closed

    def close(self):
        if self.log is not None:
            try:
                self.log.flush()
                self.log.close()
            except (ValueError, AttributeError):
                pass  # Already closed
            finally:
                self.log = None