import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from robotcord import RobotClient, extract_token

seen = []

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append((self.path, self.headers["Authorization"], body))
        self.send_response(204); self.end_headers()
    def log_message(self, *a): pass

def test_press_and_cooldown():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = RobotClient("tok", api_base=f"http://127.0.0.1:{srv.server_port}")
    assert c.press("X+").ok
    path, auth, body = seen[-1]
    assert path == "/interactions" and auth == "tok"
    assert body["data"]["custom_id"] == "move:x_plus" and body["message_id"] == c.message_id
    assert c.press("Y+").skipped == "cooldown"
    assert c.press("Abort").ok          # bypasses cooldown
    assert c.press("Safe Abort").ok
    assert seen[-1][2]["message_id"] == c.safe_message_id
    srv.shutdown()

def test_extract():
    assert extract_token("curl x -H 'authorization: abc.def'") == "abc.def"
    assert extract_token('{"authorization": "zzz"}') == "zzz"
    assert extract_token('"plain"') == "plain"

def test_dry():
    assert RobotClient("", dry_run=True).press("Zoom+").skipped == "dry_run"
