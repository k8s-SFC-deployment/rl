import time
import random
import requests
from typing import List
from src.utils import set_interval, set_timeout

class TrafficGenerator:
    files = ["files/bigFile", "files/midFile", "files/smallFile"]
    file_probs = [0.1, 0.3, 0.6]

    def __init__(self, target_url):
        self.target_url = target_url
        self.done = True

    # generate 과정에서 에러가 생길 수 있음을 고려할 것.
    def generate(self, key: str, sfs: List[str], traffic_intervals=[1], duration=60):
        if len(traffic_intervals) == 0:
            return SyntaxError("`traffic_intervals` need at least one interval.")
        self.done = False
        self.wait_times = traffic_intervals
        self.wait_time_idx = 0

        if duration != -1:
            interval_thread = set_interval(self._next_wait_time_idx, duration / len(self.wait_times))
            timeout_thread = set_timeout(self.finish, duration)
        
        while not self.done:
            path = ""
            for sf in sfs:
                if random.random() < 0.3:
                    path += sf + ","
            if len(path) == 0: continue
            else: path = path[:-1]
            file = self._choose_file()
            self._call_sfc_api(key, path, file)
            time.sleep(self.wait_times[self.wait_time_idx])
        if duration != -1:
            interval_thread.cancel()
            timeout_thread.cancel()

    def get_latencies(self, key):
        data = self._call_api("get", url=f"{self.target_url}/metrics/{key}")
        return data # { "latency": float, "count": int }
    
    def clear_latencies(self, key):
        self._call_api("delete", url=f"{self.target_url}/metrics/{key}")

    def _call_sfc_api(self, key, path, filename):
        url = f"{self.target_url}/start?key={key}&end_url={self.target_url}/end"
        files = {'file': (filename, open(filename), 'text/plain')}
        data = {
            'path': path
        }
        response = self._call_api("post", url=url, files=files, data=data)
        return response

    def _next_wait_time_idx(self):
        self.wait_time_idx = (self.wait_time_idx + 1) % len(self.wait_times)

    def finish(self):
        self.done = True

    def _call_api(self, method, *args, **kwargs):
        try:
            if method == "get":
                response = requests.get(*args, **kwargs)
                response = response.json()
            elif method == "post":
                response = requests.post(*args, **kwargs)
            elif method == "delete":
                response = requests.delete(*args, **kwargs)
            return response
        except requests.exceptions.ConnectionError as e:
            print("ConnectionError", e)
            time.sleep(1)
            return self._call_api(method, *args, **kwargs)
        except Exception as e:
            print("Error in requesting", e)
            time.sleep(1)
            return self._call_api(method, *args, **kwargs)

    def _choose_file(self):
        return random.choices(self.files, weights=self.file_probs, k=1)[0]
    
if __name__ == "__main__":
    tg = TrafficGenerator()
