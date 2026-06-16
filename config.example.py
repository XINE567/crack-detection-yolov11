DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "请填写自己的MySQL密码",
    "database": "crack_inspection",
    "charset": "utf8mb4",
    "autocommit": False
}

DEEPSEEK_CONFIG = {
    "api_key": "请填写你的DeepSeek API Key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
}

# 生产环境请通过环境变量配置：
# SECRET_KEY=随机长字符串
# SESSION_COOKIE_SECURE=1（启用 HTTPS 后设置）
