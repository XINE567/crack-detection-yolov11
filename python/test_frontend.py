#!/usr/bin/env python3
"""板端前端 WebSocket 连接测试"""
import sys
import time

sys.path.insert(0, "/userdata")
from frontend_client import FrontendClient, DEFAULT_FRONTEND_WS_URL

url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FRONTEND_WS_URL
print(f"测试连接: {url}")

c = FrontendClient(url, 5)
c.start()
try:
    for i in range(20):
        time.sleep(1)
        f = c.get_frame()
        print(
            f"{i}s streaming={c.is_streaming()} live={c.is_connected()} "
            f"frames={c.get_frame_count()} err={c.get_error()} "
            f"shape={None if f is None else f.shape}",
            flush=True,
        )
        if c.get_frame_count() >= 3:
            print("测试通过：已收到画面帧")
            break
finally:
    c.stop()
