# 建筑表面裂缝智能巡检系统后端

这是一个基于 Flask + MySQL 的裂缝巡检后端项目，用于接收 RK3588/边缘设备上传的裂缝检测结果，管理巡检工单、检测图片、裂缝指标，并支持通过 DeepSeek 生成巡检结论。

## 主要功能

- 用户登录与会话管理
- 巡检工单创建、查看、删除与详情管理
- RK3588 检测结果上传接口
- 裂缝检测图片与检测指标入库
- 检测记录列表与详情页
- 工单报告页
- DeepSeek 智能巡检结论生成
- 管理员与巡检人员基础权限区分

## 项目结构

```text
backend/
├── app.py                    # Flask 主程序入口
├── config.example.py         # 配置示例文件
├── database.sql              # 初始化数据库结构
├── migration_20260614.sql    # 旧库升级脚本
├── requirements.txt          # Python 依赖
├── test_db.py                # 数据库连接测试
├── test_upload.py            # 上传接口测试脚本
├── static/                   # 静态资源
├── templates/                # 页面模板
├── uploads/                  # 上传图片保存目录，本地生成
└── utils/
    └── db.py                 # MySQL 连接工具
```

## 环境要求

- Python 3.9 或以上
- MySQL 8.0 或以上
- pip

## 快速开始

1. 创建并启用虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 安装依赖

```powershell
pip install -r requirements.txt
```

3. 初始化数据库

先登录 MySQL，然后执行初始化脚本：

```powershell
mysql -u root -p < database.sql
```

如果是在已有旧数据库上升级，再执行：

```powershell
mysql -u root -p crack_inspection < migration_20260614.sql
```

4. 创建本地配置文件

复制配置示例：

```powershell
Copy-Item config.example.py config.py
```

然后修改 `config.py` 中的 MySQL 密码、数据库名，以及可选的 DeepSeek API Key。

配置示例：

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "你的 MySQL 密码",
    "database": "crack_inspection",
    "charset": "utf8mb4",
    "autocommit": False
}

DEEPSEEK_CONFIG = {
    "api_key": "你的 DeepSeek API Key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
}
```

5. 测试数据库连接

```powershell
python test_db.py
```

6. 启动服务

```powershell
python app.py
```

默认访问地址：

```text
http://127.0.0.1:5000
```

## 默认账号

数据库初始化脚本会创建一个管理员账号：

```text
用户名：admin
密码：123456
```

首次演示或本地测试可直接使用该账号登录。正式部署时请修改默认密码。

## 常用页面

- 登录页：`/login`
- 首页：`/home`
- 工单列表：`/workorders_page`
- 新建工单：`/create_workorder`
- 检测记录：`/records_page`
- 工单报告：`/report_page/<workorder_id>`

## 设备上传接口

接口地址：

```text
POST /upload_result
```

请求格式为 `multipart/form-data`：

- `data`：JSON 字符串，包含工单、巡检人员、位置、检测时间、检测指标等信息
- 图片文件：字段名需要与 `captures` 中的 `image_filename` 保持一致

可参考 `test_upload.py` 模拟上传。

## 上传 Git 前建议

- 提交 `README.md`、`requirements.txt`、`config.example.py`、`database.sql`、`migration_20260614.sql`、源码和模板文件
- 不要提交真实的 `config.py`，里面可能包含数据库密码和 API Key
- 不要提交 `uploads/` 中的运行时上传图片
- 不要提交 `.venv/`、`__pycache__/` 等本地环境文件

如果需要保护本地配置，可以在 `.gitignore` 中启用或添加：

```text
config.py
uploads/
test_images/
```

## 备注

当前项目直接使用明文密码做本地演示，适合课程展示和原型验证。若用于正式环境，建议改为密码哈希存储，并通过环境变量管理数据库密码、`SECRET_KEY` 和 DeepSeek API Key。
