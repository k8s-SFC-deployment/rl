import json
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class ThreadedHTTPServer(HTTPServer):
    def __init__(self, server_address, handler_class, main_queue):
        super().__init__(server_address, handler_class)
        self.main_queue = main_queue

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/step':
            # 요청 헤더를 읽고, Content-Length 확인
            content_length = int(self.headers['Content-Length'])
            # 요청 데이터 읽기
            post_data = self.rfile.read(content_length)

            # JSON 데이터 파싱
            request_data = json.loads(post_data.decode('utf-8'))

            # 요청을 메인 스레드로 전달
            request_id = threading.get_ident()
            response_queue = queue.Queue()
            self.server.main_queue.put((request_id, request_data, response_queue))

            # 메인 스레드로부터 응답을 기다림
            response_data = response_queue.get()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            del response_queue
        else:
            # 경로가 /step이 아닌 경우 404 에러 반환
            self.send_error(404, 'File Not Found: %s' % self.path)


class BackgroundHTTPServer:
    def __init__(self):
        self.main_queue = queue.Queue()

    
    def wait_request(self):
        request_id, req_data, response_queue = self.main_queue.get()
        return req_data, response_queue
    
    def send_response(self, response_data, response_queue):
        response_queue.put(response_data)
        
    def close(self):
        if hasattr(self, "httpd") and self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def run_server(self):
        server_address = ('', 8000)
        self.httpd = ThreadedHTTPServer(server_address, RequestHandler, self.main_queue)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, args=())
        self.server_thread.daemon = True
        self.server_thread.start()

if __name__ == '__main__':
    server = BackgroundHTTPServer()

    while True:
        req_data, response_queue = server.wait_request()
        print(f"Processing request with data: {req_data}")

        # 요청 처리 로직
        response_data = [1.5, 2.5, 4.5]

        server.send_response(response_data)
