"""
RK3588 WebSocket 客户端 — 连接树莓派 qianduan 接收实时画面
"""

import asyncio
import base64
import json
import os
import socket
import struct
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


DEFAULT_FRONTEND_WS_URL = os.environ.get(
    "FRONTEND_WS_URL",
    "ws://192.168.73.132:8765"
)
DEFAULT_STREAM_FPS = int(os.environ.get("FRONTEND_FPS", "5"))
RECONNECT_INTERVAL = 3.0


class _StdlibWebSocket:
    """纯标准库 WebSocket 客户端，支持分片大帧"""

    def __init__(self, url):
        parsed = urlparse(url)
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        self.path = parsed.path or "/"
        self.sock = None
        self.closed = False

    def connect(self, timeout=10):
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.settimeout(60)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket 握手失败")
            resp += chunk
        if b" 101 " not in resp:
            raise ConnectionError(resp.decode("utf-8", errors="ignore")[:300])

    def send(self, text):
        self._send_frame(0x1, text.encode("utf-8"))

    def _send_frame(self, opcode, data):
        mask = os.urandom(4)
        frame = bytearray()
        frame.append(0x80 | opcode)
        length = len(data)
        if length <= 125:
            frame.append(0x80 | length)
        elif length <= 65535:
            frame.append(0x80 | 126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack("!Q", length))
        frame.extend(mask)
        frame.extend(bytearray(b ^ mask[i % 4] for i, b in enumerate(data)))
        self.sock.sendall(frame)

    def _read_frame_payload(self, b2):
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else None
        payload = self._recv_exact(length)
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return payload

    def recv(self):
        parts = []

        while True:
            header = self._recv_exact(2)
            b1, b2 = header[0], header[1]
            fin = bool(b1 & 0x80)
            opcode = b1 & 0x0F
            payload = self._read_frame_payload(b2)

            if opcode == 0x8:
                self.closed = True
                raise ConnectionError("服务端关闭连接")

            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue

            if opcode == 0xA:
                continue

            if opcode == 0x1:
                parts = [payload]
            elif opcode == 0x0:
                parts.append(payload)
            else:
                continue

            if fin:
                return b"".join(parts).decode("utf-8")

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                self.closed = True
                raise ConnectionError("连接断开")
            buf += chunk
        return buf

    def close(self):
        self.closed = True
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


class FrontendClient:
    def __init__(self, url=None, fps=None):
        self.url = url or DEFAULT_FRONTEND_WS_URL
        self.fps = fps or DEFAULT_STREAM_FPS
        self._lock = threading.Lock()
        self._frame = None
        self._connected = False
        self._running = False
        self._thread = None
        self._ws = None
        self._last_error = ""
        self._frame_count = 0

    def start(self):
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        print(f"[Frontend] 连接 {self.url} ...", flush=True)
        return True

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        with self._lock:
            self._connected = False
        print("[Frontend] 已断开", flush=True)

    def is_connected(self):
        with self._lock:
            return self._connected and self._frame is not None

    def is_streaming(self):
        with self._lock:
            return self._connected

    def get_error(self):
        return self._last_error

    def get_frame_count(self):
        with self._lock:
            return self._frame_count

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _thread_main(self):
        while self._running:
            try:
                self._session_sync()
            except Exception as e:
                self._last_error = str(e)
                print(f"[Frontend] 连接异常: {e}", flush=True)
            with self._lock:
                self._connected = False
            if self._running:
                print(f"[Frontend] {RECONNECT_INTERVAL}s 后重连...", flush=True)
                time.sleep(RECONNECT_INTERVAL)

    def _decode_frame(self, payload):
        img_b64 = payload.get("data", {}).get("image")
        if not img_b64:
            self._last_error = "frame 消息缺少 image 字段"
            return False
        try:
            img_bytes = base64.b64decode(img_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                self._last_error = "JPEG 解码失败"
                return False
            with self._lock:
                self._frame = frame
                self._frame_count += 1
                if self._frame_count == 1 or self._frame_count % 30 == 0:
                    print(f"[Frontend] 收到帧 #{self._frame_count} {frame.shape[1]}x{frame.shape[0]}", flush=True)
            return True
        except Exception as e:
            self._last_error = f"解码失败: {e}"
            return False

    def _handle_message(self, message):
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as e:
            self._last_error = f"JSON 解析失败: {e}, len={len(message)}"
            print(f"[Frontend] {self._last_error}", flush=True)
            return

        cmd = payload.get("command")
        if cmd == "frame":
            with self._lock:
                self._connected = True
            self._decode_frame(payload)
        elif cmd == "start_stream" and payload.get("status") == "success":
            with self._lock:
                self._connected = True
            print(f"[Frontend] 推流已开始, fps={self.fps}", flush=True)
        elif payload.get("status") == "error":
            self._last_error = payload.get("message", "unknown error")
            print(f"[Frontend] 服务端错误: {self._last_error}", flush=True)

    def _session_sync(self):
        ws = _StdlibWebSocket(self.url)
        ws.connect()
        self._ws = ws
        self._last_error = ""
        ws.send(json.dumps({"command": "start_stream", "params": {"fps": self.fps}}))

        while self._running:
            message = ws.recv()
            self._handle_message(message)
            if ws.closed:
                break
