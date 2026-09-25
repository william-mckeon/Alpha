"""Bounded stdlib HTTP transport for private Phase 1 services."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException
import hmac
import socket
import time
import uuid
import json
import os
from baby_arcus.audit import TRACE
import urllib.error
import urllib.request
from baby_arcus.contracts import canonical, decode, ContractError, MAX_BODY
from baby_arcus.artifacts import Conflict, CorruptArtifact

class StaticResponse:
    def __init__(self,data,content_type):
        self.data,self.content_type=data,content_type

class RemoteError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)

class Client:
    def __init__(self, base_url, token="", timeout=5.0, attempts=2):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.attempts = attempts
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path, body=None):
        raw = canonical(body) if body is not None else None
        if raw is not None and len(raw) > MAX_BODY:
            raise ContractError("Request too large")
        deadline = time.monotonic() + self.timeout
        outgoing_trace=TRACE.get() or uuid.uuid4().hex
        last = None
        for _ in range(self.attempts):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            request = urllib.request.Request(self.base_url+path, data=raw, method=method,
                       headers={"Content-Type":"application/json","Authorization":"Bearer "+self.token,
                                "X-Arcus-Trace":outgoing_trace})
            try:
                with self.opener.open(request, timeout=remaining) as response:
                    data = response.read(MAX_BODY+1)
                    if len(data) > MAX_BODY:
                        raise RemoteError(502,"Response too large")
                    return decode(data)
            except urllib.error.HTTPError as exc:
                try:
                    last = RemoteError(exc.code, exc.read(4096).decode("utf-8","replace"))
                finally:
                    exc.close()
                if exc.code not in (502,503,504):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, HTTPException) as exc:
                last = RemoteError(503,"Service unavailable or request deadline exceeded")
        raise last or RemoteError(504,"Request deadline exceeded")

def serve(host, port, application, token="", readonly_network=False, audit=None, application_auth=False):
    if host not in ("127.0.0.1","localhost","::1") and not token and not readonly_network and not application_auth:
        raise ContractError("Non-loopback binding requires BABY_ARCUS_TOKEN")
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(10)
            self.audit_started=time.monotonic()
            self.audit_body=None
            self.audit_error=None
            self.audit_replied=False

        def log_message(self, *args):
            pass

        def handle_command(self):
            incoming=self.headers.get("X-Arcus-Trace","")
            trace=incoming if len(incoming)==32 and all(c in "0123456789abcdef" for c in incoming) else uuid.uuid4().hex
            TRACE.set(trace)
            try:
                if readonly_network and self.command!="GET":
                    return self.reply(405,{"error":"Read-only viewer"})
                if token and not hmac.compare_digest(self.headers.get("Authorization",""),"Bearer "+token):
                    return self.reply(401,{"error":"Unauthorized"})
                if self.headers.get("Transfer-Encoding"):
                    raise ContractError("Chunked requests are unsupported")
                length = int(self.headers.get("Content-Length","0"))
                if length < 0 or length > MAX_BODY:
                    return self.reply(413,{"error":"Request too large"})
                body = None
                if self.command == "POST":
                    if not self.headers.get("Content-Type","").startswith("application/json"):
                        raise ContractError("Expected application/json")
                    body = decode(self.rfile.read(length))
                self.audit_body=body
                if self.command == 'POST' and os.environ.get('ALPHA_RUNTIME_PROFILE') == 'alpha-container-v1':
                    print(json.dumps({'event':'http.request','time':time.time(),'trace':trace,
                        'port':self.server.server_port,'method':self.command,'path':self.path.split('?')[0]}),flush=True)
                if audit:
                    audit.emit("http.request",{"method":self.command,"path":self.path.split("?")[0],
                               "port":self.server.server_port,
                               "request":body},durable=self.command=="POST")
                status, result = application(self.command,self.path,body)
                self.reply(status,result)
            except Conflict as exc:
                self.reply(409,{"error":str(exc)})
            except (ContractError, ValueError, TypeError) as exc:
                self.reply(400,{"error":str(exc)})
            except KeyError:
                self.reply(404,{"error":"Not found"})
            except RemoteError:
                self.reply(503,{"error":"Dependency unavailable; retry the same request ID"})
            except CorruptArtifact:
                self.reply(500,{"error":"Artifact integrity failure"})
            except (OSError, RuntimeError) as exc:
                self.audit_error=type(exc).__name__
                self.reply(500,{"error":"Service failed; inspect local state before retry"})
            except Exception as exc:
                self.audit_error=type(exc).__name__
                self.reply(500,{"error":"Internal service error"})

        def reply(self, status, result):
            if (self.command == 'POST' or status >= 400) and os.environ.get('ALPHA_RUNTIME_PROFILE') == 'alpha-container-v1':
                print(json.dumps({'event':'http.response','time':time.time(),'trace':TRACE.get(),
                    'port':self.server.server_port,'status':status,'exception_type':self.audit_error,
                    'duration_ms':round((time.monotonic()-self.audit_started)*1000,3)}),flush=True)
            if audit and not self.audit_replied:
                self.audit_replied=True
                audit.emit("http.response",{"method":self.command,"path":self.path.split("?")[0],
                    "port":self.server.server_port,
                    "status":status,"request_id":self.audit_body.get("request_id") if isinstance(self.audit_body,dict) else None,
                    "duration_ms":round((time.monotonic()-self.audit_started)*1000,3),
                    "exception_type":self.audit_error,
                    "response":{"static_bytes":len(result.data)} if isinstance(result,StaticResponse) else result},
                    durable=self.command=="POST" or status>=400)
            data = result.data if isinstance(result,StaticResponse) else canonical(result)
            try:
                self.send_response(status)
                self.send_header("Content-Type",result.content_type if isinstance(result,StaticResponse) else "application/json")
                self.send_header("X-Content-Type-Options","nosniff")
                self.send_header("Cache-Control","no-store")
                if audit:
                    self.send_header("X-Arcus-Audit","healthy" if audit.error is None else "degraded")
                self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
                self.send_header("Content-Length",str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except ConnectionError:
                pass

        do_GET = handle_command
        do_POST = handle_command
    server = ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads = True
    if audit:audit.emit("service.listening",{"host":host,"port":server.server_port},durable=True)
    return server
