#!/usr/bin/env python3
"""
板端批量上传联调测试
用法:
  python3 test_upload_inspection.py              # 上传最新未上传会话
  python3 test_upload_inspection.py 2            # 上传 session_id=2
  python3 test_upload_inspection.py --dry-run 2  # 只打印 JSON 不上传
"""

import json
import os
import sys
import urllib.request

from inspection_db import (
    DB_PATH,
    get_session,
    get_session_captures,
    mark_capture_uploaded,
    init_db,
)
from upload_client import (
    build_inspection_payload,
    upload_inspection_session,
    get_upload_url,
    DEFAULT_BACKEND_URL,
)


def get_latest_session_id():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def main():
    init_db()
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]

    session_id = int(args[0]) if args else get_latest_session_id()
    if not session_id:
        print("没有巡检会话，请先运行主程序拍照或执行 prepare_local_test.py")
        return 1

    backend_url = os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)
    upload_url = get_upload_url(backend_url)

    print(f"Backend: {upload_url}")
    print(f"Session ID: {session_id}")

    session = get_session(session_id)
    if not session:
        print(f"会话 {session_id} 不存在")
        return 1

    captures = get_session_captures(session_id)
    pending = [c for c in captures if not c.get("uploaded")]
    print(f"总拍照: {len(captures)}, 待上传: {len(pending)}")

    if not pending:
        print("没有待上传记录")
        return 1

    payload = build_inspection_payload(session, captures)
    print("\n===== 上传 JSON =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if dry_run:
        print("\n(dry-run 模式，未实际上传)")
        return 0

    base_url = upload_url.replace("/upload_result", "")
    try:
        with urllib.request.urlopen(base_url + "/", timeout=5) as resp:
            print(f"\n连通性 OK: {resp.read().decode()[:100]}")
    except Exception as e:
        print(f"\n连通性失败: {e}")
        return 1

    saved, messages, summary, uploaded_ids, resp = upload_inspection_session(
        session, captures, backend_url
    )
    print(f"\n结果: {summary}")
    for line in messages:
        print(line)

    if saved > 0 and resp:
        for cid in uploaded_ids:
            mark_capture_uploaded(cid)
        print(f"PC 响应: code={resp.get('code')} session_id={resp.get('session_id')} "
              f"saved_count={resp.get('saved_count')} image_ids={resp.get('image_ids')}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
