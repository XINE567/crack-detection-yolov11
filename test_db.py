from utils.db import get_connection

conn = get_connection()
print("MySQL 连接成功！")
conn.close()