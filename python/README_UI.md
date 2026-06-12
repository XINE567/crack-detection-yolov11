# RK3588 UI开发指南

## 设备信息

### 屏幕规格
- **分辨率**: 1080x1920（竖屏）
- **触摸屏**: 电容式触摸屏
- **触摸设备**: `/dev/input/event2`

### 数据库
- **SQLite3** - Python内置支持
- **版本**: 3.39.2
- **使用方式**: 
```python
import sqlite3
conn = sqlite3.connect('/userdata/inspection.db')
```

## 文件说明

- `touch_handler.py` - 触摸处理模块（无需修改，直接使用）
- `ui_template.py` - UI开发模板示例
- `crack_inspection_full.py` - 完整功能示例

## 快速开始

### 1. 导入触摸模块

```python
from touch_handler import TouchHandler, create_button
```

### 2. 初始化触摸处理器

```python
touch = TouchHandler()  # 默认使用 /dev/input/event2
touch.open()
touch.start()
```

### 3. 创建按钮

```python
buttons = [
    create_button("START", 80, 1150, 500, 1280, (0, 255, 0)),
    create_button("SAVE", 560, 1150, 1000, 1280, (255, 165, 0)),
]
```

参数说明：
- `name`: 按钮名称（用于识别点击）
- `x1, y1`: 左上角坐标
- `x2, y2`: 右下角坐标
- `color`: 按钮颜色 (R, G, B)

### 4. 在主循环中检测点击

```python
last_pressed = False

while True:
    # 你的UI绘制代码...
    
    # 检测触摸点击
    if touch.is_pressed() and not last_pressed:
        btn_name = touch.check_button_click(buttons)
        if btn_name:
            # 处理按钮点击
            if btn_name == "START":
                # 执行START功能
                pass
            elif btn_name == "SAVE":
                # 执行SAVE功能
                pass
    
    last_pressed = touch.is_pressed()
    
    # 退出检测
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC键
        break

# 清理
touch.stop()
cv2.destroyAllWindows()
```

## 数据库使用示例

### 创建数据库表

```python
import sqlite3
from datetime import datetime

# 连接数据库
conn = sqlite3.connect('/userdata/inspection.db')
cursor = conn.cursor()

# 创建检测记录表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        inspector TEXT NOT NULL,
        location TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        confidence REAL,
        has_crack INTEGER,
        image_path TEXT
    )
''')
conn.commit()
```

### 保存检测记录

```python
def save_result(inspector, location, confidence, has_crack, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO inspections (inspector, location, timestamp, confidence, has_crack, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (inspector, location, timestamp, confidence, 1 if has_crack else 0, image_path))
    conn.commit()
```

### 查询历史记录

```python
def get_history(limit=10):
    cursor.execute('''
        SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    return cursor.fetchall()
```

## API文档

### TouchHandler类

#### 初始化
```python
touch = TouchHandler(device_path="/dev/input/event2")
```

#### 方法

- `open()` - 打开触摸设备，返回True/False
- `start()` - 启动触摸事件监听线程
- `stop()` - 停止监听并关闭设备
- `get_position()` - 获取当前触摸坐标 (x, y)
- `is_pressed()` - 获取当前按压状态 True/False
- `check_button_click(buttons)` - 检测是否点击了按钮，返回按钮名称或None

### Button类

#### 创建按钮
```python
btn = create_button(name, x1, y1, x2, y2, color)
```

#### 属性
- `btn.name` - 按钮名称
- `btn.x1, btn.y1` - 左上角坐标
- `btn.x2, btn.y2` - 右下角坐标
- `btn.color` - 按钮颜色

## 注意事项

### ⚠️ 重要

1. **必须使用英文界面** - 设备缺少中文字体，中文会显示乱码
2. **触摸坐标范围** - (0, 0) 到 (1080, 1920)
3. **多线程** - 触摸事件在独立线程中处理，不会阻塞UI
4. **退出清理** - 程序退出前必须调用 `touch.stop()`
5. **数据库路径** - 建议使用 `/userdata/` 目录存储数据库

### ✅ 推荐做法

1. **大按钮** - 按钮区域至少 100x100 像素
2. **清晰布局** - 按钮之间留有足够间距
3. **状态反馈** - 点击后给用户视觉反馈
4. **全屏显示** - 使用 `cv2.WINDOW_FULLSCREEN`
5. **使用SQLite** - 轻量级，无需额外安装

### ❌ 避免使用

1. **中文字符** - 会导致乱码
2. **cv2.setMouseCallback** - 无法获取触摸焦点
3. **PyQt/tkinter** - 可能未安装或不兼容
4. **阻塞操作** - 在主线程中执行耗时操作
5. **MySQL/PostgreSQL** - 需要额外安装，SQLite足够使用

## 完整示例

参考 `ui_template.py` 和 `crack_inspection_full.py` 获取完整示例。

## 部署

1. 将Python文件推送到设备：
```bash
adb push your_ui.py /userdata/your_ui.py
adb push touch_handler.py /userdata/touch_handler.py
```

2. 运行：
```bash
adb shell "cd /userdata && python3 your_ui.py"
```

## 环境要求

- Python 3
- OpenCV (cv2)
- numpy
- sqlite3（Python内置）
- Linux输入设备访问权限（通常需要root）