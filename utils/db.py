import pymysql
from config import DB_CONFIG


def get_connection():
    """
    获取 MySQL 数据库连接，并确保事务与字符集行为一致。
    """
    config = {**DB_CONFIG, "charset": "utf8mb4", "autocommit": False}
    return pymysql.connect(**config)
