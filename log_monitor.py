import os
import re
import time

def parse_log_line(line):
    pattern = r'\d+\.\d+\.\d+\.\d+ - - \[\d+/\w+/\d+ \d+:\d+:\d+\] "GET /inference/(\w+)/(\w+) HTTP/1\.1" 200 -'
    match = re.search(pattern, line)
    if match:
        token, timeframe = match.groups()
        print(f"Inference request detected: Token={token}, Timeframe={timeframe}")
    return None

def monitor_logs():
    log_file = '/app/logs/inference.log'
    
    print("Starting log monitoring...")
    while not os.path.exists(log_file):
        print("Waiting for log file to be created...")
        time.sleep(5)

    with open(log_file, 'r') as f:
        f.seek(0, 2)  # Go to the end of the file
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)  # Sleep briefly
                continue
            parse_log_line(line)

if __name__ == "__main__":
    monitor_logs()
