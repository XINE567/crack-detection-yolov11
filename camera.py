#!/usr/bin/env python3
# 树莓派端 - WebSocket 服务端 (持续推流模式)

import asyncio
import websockets
import json
import base64
import cv2
import time
import numpy as np
from picamera2 import Picamera2

# ========== 配置区域 ==========
HOST = '0.0.0.0'
PORT = 8765
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
STREAM_FPS = 5
JPEG_QUALITY = 85
SEND_WIDTH = 640
SEND_HEIGHT = 360

# ========== 初始化摄像头 ==========
print("正在初始化摄像头...")
picam2 = Picamera2()

# 使用 RGB888 格式（摄像头原生输出）
camera_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
print(f"摄像头就绪，分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# ========== 全局变量 ==========
streaming = False

# ========== 图像处理函数 ==========
def capture_and_preprocess():
    """采集图像、预处理、返回 base64 编码的图像"""
    # 采集图像（RGB 格式）
    frame = picam2.capture_array()
    
    # 不需要颜色转换，直接缩放
    resized = cv2.resize(frame, (SEND_WIDTH, SEND_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 编码为 JPEG
    _, img_encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    # 转换为 Base64
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    
    return img_base64

# ========== WebSocket 客户端处理 ==========
async def handle_client(websocket):  # 移除 path 参数
    global streaming
    print(f"[连接] 客户端已连接: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                req = json.loads(message)
                cmd = req.get('command')
                params = req.get('params', {})
                
                print(f"[指令] 收到: {cmd}")
                
                # 开始推流
                if cmd == 'start_stream':
                    streaming = True
                    fps = params.get('fps', STREAM_FPS)
                    interval = 1.0 / fps
                    
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'command': 'start_stream',
                        'data': {'fps': fps}
                    }))
                    print(f"[推流] 开始推流，帧率: {fps} fps")
                    
                    while streaming:
                        img_base64 = capture_and_preprocess()
                        await websocket.send(json.dumps({
                            'command': 'frame',
                            'data': {'image': img_base64}
                        }))
                        await asyncio.sleep(interval)
                
                # 停止推流
                elif cmd == 'stop_stream':
                    streaming = False
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'command': 'stop_stream'
                    }))
                    print("[推流] 停止推流")
                
                # 心跳检测
                elif cmd == 'ping':
                    await websocket.send(json.dumps({
                        'status': 'success',
                        'command': 'ping',
                        'data': {'pong': True}
                    }))
                    print("[心跳] 已响应 pong")
                
                # 未知指令
                else:
                    await websocket.send(json.dumps({
                        'status': 'error',
                        'command': cmd,
                        'message': f'未知指令: {cmd}'
                    }))
                    print(f"[错误] 未知指令: {cmd}")
                    
            except json.JSONDecodeError:
                print(f"[错误] 无效 JSON")
                await websocket.send(json.dumps({
                    'status': 'error',
                    'message': '无效的 JSON 格式'
                }))
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[断开] 客户端断开: {websocket.remote_address}")
    finally:
        streaming = False
        print(f"[清理] 连接已关闭")

# ========== 主函数 ==========
async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print("=" * 50)
        print("树莓派 WebSocket 服务端 (持续推流模式)")
        print(f"监听地址: ws://{HOST}:{PORT}")
        print("等待 RK3588 连接...")
        print("按 Ctrl+C 停止")
        print("=" * 50)
        await asyncio.Future()

# ========== 程序入口 ==========
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务已停止")
        picam2.stop()
        cv2.destroyAllWindows()