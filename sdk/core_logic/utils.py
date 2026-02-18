
import threading
import queue
import sys
import datetime

class Logger:
    def __init__(self, queue):
        self.queue = queue

    def log(self, message, level='info'):
        if self.queue:
            self.queue.put((message, level))
        else:
            print(f"[{level.upper()}] {message}")

    def info(self, message):
        self.log(message, 'info')

    def error(self, message):
        self.log(message, 'error')

    def success(self, message):
        self.log(message, 'success')
    
    def progress(self, message):
        self.log(message, 'progress')

class TeeIO:
    """Redirect stdout/stderr to a queue."""
    def __init__(self, original_stream, queue, level='info'):
        self.original_stream = original_stream
        self.queue = queue
        self.level = level

    def write(self, s):
        if self.original_stream:
            self.original_stream.write(s)
        
        # Simple filtering to avoid sending empty newlines as separate messages
        if s.strip():
            # Check for tqdm-like progress bars
            if '%' in s and '|' in s and ('it/s' in s or 's/it' in s):
                 self.queue.put((s.strip(), 'progress'))
            else:
                 self.queue.put((s.strip(), self.level))

    def flush(self):
        if self.original_stream:
            self.original_stream.flush()

    def isatty(self):
        return self.original_stream.isatty() if hasattr(self.original_stream, "isatty") else False

def setup_process_logging(log_queue):
    """Redirects stdout and stderr to the log queue."""
    sys.stdout = TeeIO(sys.stdout, log_queue, 'info')
    sys.stderr = TeeIO(sys.stderr, log_queue, 'error')

