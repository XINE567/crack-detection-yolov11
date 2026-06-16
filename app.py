import os
import json
import pymysql
import secrets
import uuid
import requests
from functools import wraps
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import DEEPSEEK_CONFIG
from utils.db import get_connection


app = Flask(__name__)
if hasattr(app, "json"):
    app.json.ensure_ascii = False
else:
    app.config["JSON_AS_ASCII"] = False
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE") == "1"
CORS(app, resources={
    r"/upload_result": {"origins": "*"},
    r"/device/workorders": {"origins": "*"},
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/records"):
                return jsonify({"code": 401, "message": "请先登录"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if session.get("role") != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def can_access_workorder(user_id):
    if session.get("role") == "admin":
        return True
    return user_id == session.get("user_id")


def safe_number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def request_deepseek_conclusion(workorder, records):
    api_key = str(DEEPSEEK_CONFIG.get("api_key") or "").strip()
    base_url = str(DEEPSEEK_CONFIG.get("base_url") or "https://api.deepseek.com").rstrip("/")
    model = str(DEEPSEEK_CONFIG.get("model") or "deepseek-chat").strip()
    if not api_key:
        raise ValueError("请先在 config.py 的 DEEPSEEK_CONFIG 中填写 api_key")

    inspection_data = {
        "任务ID": workorder["workorder_id"],
        "巡检位置": workorder["location"],
        "任务说明": workorder.get("description"),
        "检测记录": [
            {
                "采集ID": record.get("capture_id"),
                "裂纹数量": record.get("crack_count"),
                "裂纹面积": record.get("crack_area"),
                "最大宽度": record.get("max_width"),
                "总长度": record.get("total_length"),
                "严重等级": record.get("severity"),
                "置信度": record.get("confidence"),
                "现有建议": record.get("suggestion"),
            }
            for record in records
        ],
    }
    prompt = (
        "你是建筑表面裂纹巡检专家。请根据以下机器检测结果生成中文巡检结论。"
        "结论应包含总体风险判断、关键异常、建议处置措施和后续复检建议。"
        "不要虚构检测数据，不要声称已经直接看过图片，控制在500字以内。\n"
        + json.dumps(inspection_data, ensure_ascii=False, default=str)
    )
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "thinking": {"type": "enabled", "reasoning_effort": "max"},
            "messages": [
                {"role": "system", "content": "你负责根据裂纹检测数据生成严谨、可执行的巡检结论。"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=90,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip(), model


def parse_form_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def stored_filename(original_name, prefix="result"):
    safe_name = secure_filename(original_name)
    extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "jpg"
    return f"{datetime.now():%Y%m%d}_{prefix}_{uuid.uuid4().hex}.{extension}"


def remove_files(paths):
    for path in paths:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except OSError:
            app.logger.exception("清理文件失败: %s", path)


@app.context_processor
def inject_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["csrf_token"]}


@app.before_request
def validate_csrf():
    if request.method == "POST" and request.endpoint != "upload_result":
        supplied = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not expected or not secrets.compare_digest(supplied, expected):
            abort(400, description="请求校验失败，请刷新页面后重试")


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"code": 413, "message": "上传内容超过大小限制"}), 413


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "code": 200,
        "message": "建筑表面裂纹智能巡检系统 Flask 后端已启动"
    })


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    用户登录页面。
    """

    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password")

    if not username or not password:
        return render_template("login.html", error="请输入用户名和密码")

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT id, username, password, role
                FROM users
                WHERE username = %s
            """
            cursor.execute(sql, (username,))
            user = cursor.fetchone()

            valid_password = bool(user and user["password"] == password)
    finally:
        conn.close()

    if not user or not valid_password:
        return render_template("login.html", error="用户名或密码错误"), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]

    next_url = request.args.get("next", "")
    return redirect(next_url if next_url.startswith("/") and not next_url.startswith("//") else url_for("home"))


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/home", methods=["GET"])
@login_required
def home():
    """
    系统统一首页。
    不再区分管理员和操作人员，所有登录用户进入同一管理界面。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            owner_sql = "" if session.get("role") == "admin" else " WHERE user_id = %s"
            owner_params = () if session.get("role") == "admin" else (session["user_id"],)

            cursor.execute(f"SELECT COUNT(*) AS total FROM workorders{owner_sql}", owner_params)
            total_orders = cursor.fetchone()["total"]

            cursor.execute(
                f"SELECT COUNT(*) AS total FROM images img JOIN workorders w ON img.workorder_id = w.id"
                + ("" if session.get("role") == "admin" else " WHERE w.user_id = %s"),
                owner_params
            )
            total_images = cursor.fetchone()["total"]

            cursor.execute(
                "SELECT COUNT(*) AS total FROM crack_results r "
                "JOIN images img ON r.image_id = img.id JOIN workorders w ON img.workorder_id = w.id "
                "WHERE r.severity IN ('中等', '严重', 'MEDIUM', 'HIGH')"
                + ("" if session.get("role") == "admin" else " AND w.user_id = %s"),
                owner_params
            )
            abnormal_count = cursor.fetchone()["total"]

            cursor.execute(
                "SELECT COUNT(*) AS total FROM workorders WHERE status = '已完成'"
                + ("" if session.get("role") == "admin" else " AND user_id = %s"),
                owner_params
            )
            finished_orders = cursor.fetchone()["total"]

    finally:
        conn.close()

    return render_template(
        "home.html",
        username=session.get("username"),
        total_orders=total_orders,
        total_images=total_images,
        abnormal_count=abnormal_count,
        finished_orders=finished_orders
    )

@app.route("/admin_home", methods=["GET"])
def admin_home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        return redirect(url_for("inspector_home"))

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM workorders")
            total_orders = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS total FROM images")
            total_images = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS total FROM crack_results WHERE severity IN ('中等', '严重')")
            abnormal_count = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS total FROM workorders WHERE status = '已完成'")
            finished_orders = cursor.fetchone()["total"]

    finally:
        conn.close()

    return render_template(
        "admin_home.html",
        username=session.get("username"),
        total_orders=total_orders,
        total_images=total_images,
        abnormal_count=abnormal_count,
        finished_orders=finished_orders
    )


@app.route("/inspector_home", methods=["GET"])
def inspector_home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    id AS workorder_id,
                    location,
                    status,
                    created_at,
                    completed_at
                FROM workorders
                WHERE user_id = %s
                ORDER BY created_at DESC
            """
            cursor.execute(sql, (user_id,))
            workorders = cursor.fetchall()

    finally:
        conn.close()

    return render_template(
        "inspector_home.html",
        username=session.get("username"),
        workorders=workorders
    )


@app.route("/uploads/<path:filename>", methods=["GET"])
@login_required
def uploaded_file(filename):
    """
    访问上传后的图片。
    例如：http://127.0.0.1:5000/uploads/20260510_test.jpg
    """
    safe_name = os.path.basename(filename)
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                """
                SELECT w.user_id
                FROM images img
                JOIN workorders w ON img.workorder_id = w.id
                WHERE img.image_path LIKE %s OR img.original_image_path LIKE %s
                LIMIT 1
                """,
                (f"%{safe_name}", f"%{safe_name}")
            )
            owner = cursor.fetchone()
    finally:
        conn.close()

    if not owner or not can_access_workorder(owner["user_id"]):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, safe_name)


@app.route("/upload_result", methods=["POST"])
def upload_result():
    """
    接收 RK3588 上传的一次巡检任务结果。

    请求格式：
    multipart/form-data

    字段：
    - data: JSON 字符串，包含 session_id、inspector、location、captures 等
    - 多个图片文件：字段名建议与 captures 中的 image_filename 一致
    """

    data_str = request.form.get("data")

    if not data_str:
        return jsonify({
            "code": 400,
            "message": "缺少 data 字段"
        }), 400

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return jsonify({
            "code": 400,
            "message": "data 字段不是合法 JSON"
        }), 400

    session_id = data.get("session_id") or data.get("workorder_id") or 1
    inspector = str(data.get("inspector") or "").strip()[:100]
    location = str(data.get("location") or "RK3588上传任务").strip()[:255]
    inspection_start_time = data.get("inspection_start_time")
    inspection_end_time = data.get("inspection_end_time")
    captures = data.get("captures", [])

    # 兼容原有 RK3588 单图片直传格式：image + data(workorder_id/检测指标)。
    if not captures and request.files.get("image"):
        captures = [{
            "capture_id": data.get("capture_id") or f"rk3588-{uuid.uuid4().hex[:12]}",
            "image_filename": "image",
            "original_image_filename": data.get("original_image_filename", ""),
            "captured_at": data.get("captured_at"),
            "crack_count": data.get("crack_count", 0),
            "crack_area": data.get("crack_area", 0),
            "max_width": data.get("max_width", 0),
            "total_length": data.get("total_length", 0),
            "severity": data.get("severity", "无"),
            "suggestion": data.get("suggestion", ""),
            "confidence": data.get("confidence", 0),
            "result_json": data.get("result_json", {}),
        }]

    try:
        session_id = int(session_id)
    except (TypeError, ValueError):
        session_id = 1

    if session_id < 1 or session_id > 2147483647:
        return jsonify({
            "code": 400,
            "message": "session_id 超出范围，请使用 1 到 2147483647 之间的工单编号"
        }), 400

    if not isinstance(captures, list) or not captures:
        return jsonify({
            "code": 400,
            "message": "没有收到图片；请使用 image 字段或 captures 批量格式"
        }), 400

    prepared_captures = []
    for index, capture in enumerate(captures, start=1):
        if not isinstance(capture, dict):
            continue

        capture_id = str(capture.get("capture_id") or f"rk3588-{uuid.uuid4().hex[:12]}").strip()[:100]
        image_filename = str(capture.get("image_filename") or "image").strip()
        original_image_filename = str(capture.get("original_image_filename") or "").strip()

        image_file = request.files.get(image_filename)
        original_image_file = request.files.get(original_image_filename) if original_image_filename else None
        if not image_file:
            continue
        try:
            json.dumps(capture.get("result_json", {}), ensure_ascii=False)
        except (TypeError, ValueError):
            capture["result_json"] = {}

        capture["capture_id"] = capture_id
        prepared_captures.append((capture, image_file, original_image_file))

    if not prepared_captures:
        return jsonify({"code": 400, "message": "没有收到可保存的图片文件"}), 400

    conn = get_connection()
    saved_image_ids = []
    saved_paths = []
    skipped_capture_ids = []

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            user_id = None

            if inspector:
                cursor.execute(
                    "SELECT id FROM users WHERE username = %s",
                    (inspector,)
                )
                user = cursor.fetchone()

                if user:
                    user_id = user["id"]

            cursor.execute("SELECT id, user_id FROM workorders WHERE id = %s", (session_id,))
            existing_workorder = cursor.fetchone()

            if existing_workorder:
                update_workorder_sql = """
                    UPDATE workorders
                    SET
                        user_id = COALESCE(user_id, %s),
                        inspector_name = %s,
                        location = %s,
                        status = '已完成',
                        start_time = %s,
                        end_time = %s,
                        completed_at = %s,
                        capture_count = %s
                    WHERE id = %s
                """
                cursor.execute(
                    update_workorder_sql,
                    (
                        user_id,
                        inspector,
                        location,
                        inspection_start_time,
                        inspection_end_time,
                        inspection_end_time,
                        len(prepared_captures),
                        session_id
                    )
                )
            else:
                insert_workorder_sql = """
                    INSERT INTO workorders
                    (
                        id,
                        user_id,
                        inspector_name,
                        location,
                        source,
                        status,
                        created_at,
                        start_time,
                        end_time,
                        completed_at,
                        capture_count
                    )
                    VALUES (%s, %s, %s, %s, 'device', '已完成', NOW(), %s, %s, %s, %s)
                """
                cursor.execute(
                    insert_workorder_sql,
                    (
                        session_id,
                        user_id,
                        inspector,
                        location,
                        inspection_start_time,
                        inspection_end_time,
                        inspection_end_time,
                        len(prepared_captures)
                    )
                )

            for capture, image_file, original_image_file in prepared_captures:
                capture_id = str(capture.get("capture_id")).strip()
                captured_at = capture.get("captured_at")
                image_filename = capture.get("image_filename")

                crack_count = max(0, int(safe_number(capture.get("crack_count", 0))))
                crack_area = max(0, safe_number(capture.get("crack_area", 0)))
                max_width = max(0, safe_number(capture.get("max_width", 0)))
                total_length = max(0, safe_number(capture.get("total_length", 0)))
                severity = str(capture.get("severity", "无"))[:50]
                suggestion = str(capture.get("suggestion", ""))[:500]
                confidence = min(1, max(0, safe_number(capture.get("confidence", 0))))
                result_json = capture.get("result_json", {})

                cursor.execute(
                    "SELECT id FROM images WHERE workorder_id = %s AND capture_id = %s",
                    (session_id, capture_id)
                )
                if cursor.fetchone():
                    capture_id = f"{capture_id[:87]}-{uuid.uuid4().hex[:12]}"

                saved_filename = stored_filename(image_file.filename)
                image_path = os.path.join(UPLOAD_FOLDER, saved_filename)
                image_file.save(image_path)
                saved_paths.append(image_path)

                original_image_path = None
                if original_image_file:
                    original_saved_filename = stored_filename(original_image_file.filename, "original")
                    original_image_path = os.path.join(UPLOAD_FOLDER, original_saved_filename)
                    original_image_file.save(original_image_path)
                    saved_paths.append(original_image_path)

                insert_image_sql = """
                    INSERT INTO images
                    (
                        workorder_id,
                        capture_id,
                        captured_at,
                        image_path,
                        original_image_path,
                        image_filename
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(
                    insert_image_sql,
                    (
                        session_id,
                        capture_id,
                        captured_at,
                        image_path,
                        original_image_path,
                        image_filename
                    )
                )

                image_id = cursor.lastrowid
                saved_image_ids.append(image_id)

                insert_result_sql = """
                    INSERT INTO crack_results
                    (
                        image_id,
                        crack_count,
                        crack_area,
                        max_width,
                        total_length,
                        severity,
                        suggestion,
                        confidence,
                        result_json,
                        detected_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(
                    insert_result_sql,
                    (
                        image_id,
                        crack_count,
                        crack_area,
                        max_width,
                        total_length,
                        severity,
                        suggestion,
                        confidence,
                        json.dumps(result_json, ensure_ascii=False),
                        captured_at
                    )
                )

            cursor.execute(
                """
                UPDATE workorders
                SET capture_count = (
                    SELECT COUNT(*) FROM images WHERE images.workorder_id = workorders.id
                )
                WHERE id = %s
                """,
                (session_id,)
            )

        conn.commit()

    except Exception as e:
        conn.rollback()
        remove_files(saved_paths)
        app.logger.exception("上传结果写入失败")
        return jsonify({
            "code": 500,
            "message": "上传写入失败，请稍后重试"
        }), 500

    finally:
        conn.close()

    return jsonify({
        "code": 200,
        "message": "上传成功",
        "session_id": session_id,
        "saved_count": len(saved_image_ids),
        "image_ids": saved_image_ids,
        "skipped_capture_ids": skipped_capture_ids
    })


@app.route("/device/workorders", methods=["GET"])
def device_workorders():
    """供 RK3588 获取人工创建且尚未完成的工单。"""
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    w.id AS session_id,
                    w.location,
                    w.description,
                    w.status,
                    w.created_at,
                    w.start_time,
                    w.inspector_name,
                    u.username
                FROM workorders w
                LEFT JOIN users u ON w.user_id = u.id
                WHERE w.source = 'manual'
                  AND w.status IN ('待开始', '待接收', '巡检中')
                ORDER BY w.created_at ASC
                """
            )
            workorders = cursor.fetchall()
    finally:
        conn.close()

    return jsonify({"code": 200, "workorders": workorders})


@app.route("/records", methods=["GET"])
@login_required
def get_records():
    """
    查询所有历史巡检记录。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    img.id AS image_id,
                    img.image_path,
                    img.upload_time,

                    w.id AS workorder_id,
                    w.location,
                    w.status,

                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.detected_at
                FROM images img
                LEFT JOIN workorders w
                ON img.workorder_id = w.id
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE (%s = 'admin' OR w.user_id = %s)
                ORDER BY img.upload_time DESC
            """

            cursor.execute(sql, (session.get("role"), session.get("user_id")))
            records = cursor.fetchall()

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": "查询失败",
            "error": str(e)
        }), 500

    finally:
        conn.close()

    return jsonify({
        "code": 200,
        "data": records
    })


@app.route("/records/<int:image_id>", methods=["GET"])
@login_required
def get_record_detail(image_id):
    """
    查询单条巡检记录详情。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    img.id AS image_id,
                    img.image_path,
                    img.upload_time,

                    w.id AS workorder_id,
                    w.location,
                    w.status,

                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.result_json,
                    r.detected_at
                FROM images img
                LEFT JOIN workorders w
                ON img.workorder_id = w.id
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE img.id = %s
                  AND (%s = 'admin' OR w.user_id = %s)
            """

            cursor.execute(sql, (image_id, session.get("role"), session.get("user_id")))
            record = cursor.fetchone()

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": "查询详情失败",
            "error": str(e)
        }), 500

    finally:
        conn.close()

    if not record:
        return jsonify({
            "code": 404,
            "message": "未找到该巡检记录"
        }), 404

    return jsonify({
        "code": 200,
        "data": record
    })


@app.route("/records_page", methods=["GET"])
@login_required
def records_page():
    """
    历史巡检记录网页展示。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    img.id AS image_id,
                    img.image_path,
                    img.upload_time,

                    w.id AS workorder_id,
                    w.location,
                    w.status,

                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.detected_at
                FROM images img
                LEFT JOIN workorders w
                ON img.workorder_id = w.id
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE (%s = 'admin' OR w.user_id = %s)
                ORDER BY img.upload_time DESC
            """

            cursor.execute(sql, (session.get("role"), session.get("user_id")))
            records = cursor.fetchall()

    finally:
        conn.close()

    return render_template("records.html", records=records)


@app.route("/record_page/<int:image_id>", methods=["GET"])
@login_required
def record_page(image_id):
    """
    单条巡检记录详情网页。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    img.id AS image_id,
                    img.capture_id,
                    img.image_path,
                    img.original_image_path,
                    img.image_filename,
                    img.captured_at,
                    img.upload_time,

                    w.id AS workorder_id,
                    w.location,
                    w.status,
                    w.inspector_name,
                    w.user_id,

                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.confidence,
                    r.result_json,
                    r.detected_at
                FROM images img
                LEFT JOIN workorders w
                ON img.workorder_id = w.id
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE img.id = %s
            """

            cursor.execute(sql, (image_id,))
            record = cursor.fetchone()

    finally:
        conn.close()

    if not record or not can_access_workorder(record.get("user_id")):
        return "未找到该巡检记录", 404

    image_url = url_for("uploaded_file", filename=os.path.basename(record["image_path"]))

    original_image_url = None
    if record.get("original_image_path"):
        original_image_url = url_for("uploaded_file", filename=os.path.basename(record["original_image_path"]))

    return render_template(
        "record_detail.html",
        record=record,
        image_url=image_url,
        original_image_url=original_image_url
)




@app.route("/workorders_page", methods=["GET"])
@login_required
def workorders_page():
    """
    巡检任务列表页面。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = """
                SELECT
                    w.id AS workorder_id,
                    w.location,
                    w.status,
                    w.inspector_name,
                    w.capture_count,
                    w.created_at,
                    w.start_time,
                    w.end_time,
                    w.completed_at,
                    u.username
                FROM workorders w
                LEFT JOIN users u
                ON w.user_id = u.id
                WHERE (%s = 'admin' OR w.user_id = %s)
                ORDER BY w.created_at DESC
            """

            cursor.execute(sql, (session.get("role"), session.get("user_id")))
            workorders = cursor.fetchall()

    except Exception as e:
        return f"查询巡检任务失败：{e}", 500

    finally:
        conn.close()

    return render_template(
        "workorders.html",
        workorders=workorders,
        username=session.get("username"),
        role=session.get("role")
    )



@app.route("/workorder_page/<int:workorder_id>", methods=["GET"])
@login_required
def workorder_page(workorder_id):
    """
    巡检任务详情页面：查看某个 session / 工单下的所有采集图片和裂纹检测结果。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            order_sql = """
                SELECT
                    w.id AS workorder_id,
                    w.location,
                    w.status,
                    w.source,
                    w.description,
                    w.created_at,
                    w.start_time,
                    w.end_time,
                    w.completed_at,
                    w.inspector_name,
                    w.capture_count,
                    w.user_id,
                    u.username,
                    u.role
                FROM workorders w
                LEFT JOIN users u
                ON w.user_id = u.id
                WHERE w.id = %s
            """
            cursor.execute(order_sql, (workorder_id,))
            workorder = cursor.fetchone()

            if not workorder:
                return "未找到该巡检任务", 404
            if not can_access_workorder(workorder.get("user_id")):
                return "无权查看该巡检任务", 403

            records_sql = """
                SELECT
                    img.id AS image_id,
                    img.capture_id,
                    img.image_path,
                    img.original_image_path,
                    img.image_filename,
                    img.captured_at,
                    img.upload_time,
                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.confidence,
                    r.detected_at
                FROM images img
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE img.workorder_id = %s
                ORDER BY img.captured_at DESC
            """
            cursor.execute(records_sql, (workorder_id,))
            records = cursor.fetchall()

    except Exception as e:
        return f"查询巡检任务详情失败：{e}", 500

    finally:
        conn.close()

    return render_template(
        "workorder_detail.html",
        workorder=workorder,
        records=records
    )


@app.route("/workorder_page/<int:workorder_id>/manual_upload", methods=["POST"])
@login_required
def manual_upload_workorder_record(workorder_id):
    """允许登录用户向自己可访问的工单人工添加一条检测记录。"""
    detected_image = request.files.get("detected_image")
    original_image = request.files.get("original_image")
    capture_id = request.form.get("capture_id", "").strip() or f"manual-{uuid.uuid4().hex[:12]}"
    captured_at_value = request.form.get("captured_at", "").strip()
    captured_at = parse_form_datetime(captured_at_value) or datetime.now()
    severity = request.form.get("severity", "无").strip()
    suggestion = request.form.get("suggestion", "").strip()[:500]
    mark_completed = request.form.get("mark_completed") == "1"

    if len(capture_id) > 100:
        return "采集ID不能超过100个字符", 400
    if severity not in {"无", "轻微", "中等", "严重"}:
        return "严重等级不合法", 400
    if not detected_image:
        return "请选择检测图片", 400

    capture = {
        "crack_count": request.form.get("crack_count", 0),
        "crack_area": request.form.get("crack_area", 0),
        "max_width": request.form.get("max_width", 0),
        "total_length": request.form.get("total_length", 0),
        "confidence": request.form.get("confidence", 0),
    }
    crack_count = int(safe_number(capture["crack_count"]))

    saved_paths = []
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                "SELECT user_id, source FROM workorders WHERE id = %s FOR UPDATE",
                (workorder_id,),
            )
            workorder = cursor.fetchone()
            if not workorder:
                return "未找到该巡检任务", 404
            if not can_access_workorder(workorder.get("user_id")):
                return "无权修改该巡检任务", 403
            if workorder.get("source") != "manual":
                return "仅人工新建任务支持人工上传采集与检测记录", 403

            cursor.execute(
                "SELECT id FROM images WHERE workorder_id = %s AND capture_id = %s",
                (workorder_id, capture_id),
            )
            if cursor.fetchone():
                return "该采集ID已存在，请更换后重试", 409

            detected_name = stored_filename(detected_image.filename)
            detected_path = os.path.join(UPLOAD_FOLDER, detected_name)
            detected_image.save(detected_path)
            saved_paths.append(detected_path)

            original_path = None
            if original_image and original_image.filename:
                original_name = stored_filename(original_image.filename, "original")
                original_path = os.path.join(UPLOAD_FOLDER, original_name)
                original_image.save(original_path)
                saved_paths.append(original_path)

            cursor.execute(
                """
                INSERT INTO images
                    (workorder_id, capture_id, captured_at, image_path, original_image_path, image_filename)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (workorder_id, capture_id, captured_at, detected_path, original_path, detected_image.filename),
            )
            image_id = cursor.lastrowid
            cursor.execute(
                """
                INSERT INTO crack_results
                    (image_id, crack_count, crack_area, max_width, total_length, severity,
                     suggestion, confidence, result_json, detected_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    crack_count,
                    safe_number(capture["crack_area"]),
                    safe_number(capture["max_width"]),
                    safe_number(capture["total_length"]),
                    severity,
                    suggestion,
                    min(1, max(0, safe_number(capture["confidence"]))),
                    json.dumps({"source": "manual"}, ensure_ascii=False),
                    captured_at,
                ),
            )
            cursor.execute(
                """
                UPDATE workorders
                SET
                    capture_count = (SELECT COUNT(*) FROM images WHERE workorder_id = %s),
                    status = CASE WHEN %s THEN '已完成' ELSE status END,
                    completed_at = CASE WHEN %s THEN NOW() ELSE completed_at END,
                    end_time = CASE WHEN %s THEN COALESCE(end_time, NOW()) ELSE end_time END
                WHERE id = %s
                """,
                (workorder_id, mark_completed, mark_completed, mark_completed, workorder_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        remove_files(saved_paths)
        app.logger.exception("人工上传检测记录失败")
        return "人工上传失败，请稍后重试", 500
    finally:
        conn.close()

    return redirect(url_for("workorder_page", workorder_id=workorder_id))


@app.route("/workorder_page/<int:workorder_id>/deepseek_analysis", methods=["POST"])
@login_required
def deepseek_analysis(workorder_id):
    conn = get_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                """
                SELECT id AS workorder_id, user_id, location, description
                FROM workorders
                WHERE id = %s
                """,
                (workorder_id,),
            )
            workorder = cursor.fetchone()
            if not workorder:
                return "未找到该巡检任务", 404
            if not can_access_workorder(workorder.get("user_id")):
                return "无权分析该巡检任务", 403

            cursor.execute(
                """
                SELECT img.capture_id, r.crack_count, r.crack_area, r.max_width,
                       r.total_length, r.severity, r.suggestion, r.confidence
                FROM images img
                LEFT JOIN crack_results r ON img.id = r.image_id
                WHERE img.workorder_id = %s
                ORDER BY img.captured_at ASC
                """,
                (workorder_id,),
            )
            records = cursor.fetchall()
            conclusion, model = request_deepseek_conclusion(workorder, records)
            cursor.execute(
                """
                UPDATE workorders
                SET ai_conclusion = %s, ai_model = %s, ai_analyzed_at = NOW()
                WHERE id = %s
                """,
                (conclusion, model, workorder_id),
            )
        conn.commit()
    except (ValueError, requests.RequestException, KeyError, IndexError) as error:
        conn.rollback()
        return f"DeepSeek 分析失败：{error}", 502
    except Exception as error:
        conn.rollback()
        app.logger.exception("DeepSeek 分析失败")
        return f"DeepSeek 分析失败：{error}", 500
    finally:
        conn.close()

    return redirect(url_for("report_page", workorder_id=workorder_id))


@app.route("/report_page/<int:workorder_id>", methods=["GET"])
@login_required
def report_page(workorder_id):
    """
    根据巡检任务生成网页版巡检报告。
    """

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            order_sql = """
                SELECT
                    w.id AS workorder_id,
                    w.location,
                    w.status,
                    w.description,
                    w.created_at,
                    w.start_time,
                    w.end_time,
                    w.completed_at,
                    w.inspector_name,
                    w.capture_count,
                    w.ai_conclusion,
                    w.ai_model,
                    w.ai_analyzed_at,
                    w.user_id,
                    u.username,
                    u.role
                FROM workorders w
                LEFT JOIN users u
                ON w.user_id = u.id
                WHERE w.id = %s
            """
            cursor.execute(order_sql, (workorder_id,))
            workorder = cursor.fetchone()

            if not workorder:
                return "未找到该巡检任务", 404
            if not can_access_workorder(workorder.get("user_id")):
                return "无权查看该巡检报告", 403

            records_sql = """
                SELECT
                    img.id AS image_id,
                    img.capture_id,
                    img.image_filename,
                    img.original_image_path,
                    img.original_image_path,
                    img.image_path,
                    img.captured_at,
                    img.upload_time,
                    r.crack_count,
                    r.crack_area,
                    r.max_width,
                    r.total_length,
                    r.severity,
                    r.suggestion,
                    r.confidence,
                    r.detected_at
                FROM images img
                LEFT JOIN crack_results r
                ON img.id = r.image_id
                WHERE img.workorder_id = %s
                ORDER BY img.captured_at ASC
            """
            cursor.execute(records_sql, (workorder_id,))
            records = cursor.fetchall()

    except Exception as e:
        return f"生成报告失败：{e}", 500

    finally:
        conn.close()

    image_count = len(records)
    total_crack_count = sum((r["crack_count"] or 0) for r in records)
    total_crack_area = sum((r["crack_area"] or 0) for r in records)
    total_length = sum((r["total_length"] or 0) for r in records)
    max_width = max([(r["max_width"] or 0) for r in records], default=0)

    severity_order = {
        "无": 0,
        "轻微": 1,
        "中等": 2,
        "严重": 3,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3
    }

    highest_severity = "无"
    highest_score = 0

    for r in records:
        severity = r["severity"]
        score = severity_order.get(severity, 0)
        if score > highest_score:
            highest_score = score
            highest_severity = severity

    confidence_values = [
        r["confidence"] for r in records
        if r["confidence"] is not None and r["confidence"] > 0
    ]

    avg_confidence = 0
    if confidence_values:
        avg_confidence = round(sum(confidence_values) / len(confidence_values), 3)

    return render_template(
        "report.html",
        workorder=workorder,
        records=records,
        image_count=image_count,
        total_crack_count=total_crack_count,
        total_crack_area=total_crack_area,
        total_length=total_length,
        max_width=max_width,
        highest_severity=highest_severity,
        avg_confidence=avg_confidence
    )


@app.route("/create_workorder", methods=["GET", "POST"])
@login_required
def create_workorder():
    """
    人工创建任务元数据。
    保存后可人工上传采集与检测记录，也可交由 RK3588 执行。
    """

    conn = get_connection()

    if request.method == "GET":
        try:
            with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                if session.get("role") == "admin":
                    cursor.execute("SELECT id, username, role FROM users ORDER BY id ASC")
                    users = cursor.fetchall()
                else:
                    users = []

        finally:
            conn.close()

        return render_template("create_workorder.html", users=users)

    location = request.form.get("location", "").strip()
    description = request.form.get("description", "").strip()
    user_id = request.form.get("user_id") if session.get("role") == "admin" else session.get("user_id")
    status = request.form.get("status", "待接收")
    start_time_value = request.form.get("start_time", "").strip()
    end_time_value = request.form.get("end_time", "").strip()
    start_time = parse_form_datetime(start_time_value)
    end_time = parse_form_datetime(end_time_value)

    if not location:
        conn.close()
        return "巡检位置不能为空", 400
    if len(location) > 255:
        conn.close()
        return "巡检位置不能超过 255 个字符", 400
    if status not in {"待接收", "巡检中", "已完成"}:
        conn.close()
        return "任务状态不合法", 400
    if (start_time_value and not start_time) or (end_time_value and not end_time):
        conn.close()
        return "任务时间格式不正确", 400
    if end_time and not start_time:
        conn.close()
        return "填写结束时间时必须同时填写开始时间", 400
    if start_time and end_time and end_time < start_time:
        conn.close()
        return "结束时间不能早于开始时间", 400
    if status == "已完成" and not end_time:
        conn.close()
        return "已完成任务必须填写结束时间", 400

    if not user_id:
        user_id = session.get("user_id")

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
            owner = cursor.fetchone()
            if not owner:
                return "指定的巡检人员不存在", 400
            sql = """
                INSERT INTO workorders
                (
                    user_id,
                    inspector_name,
                    location,
                    description,
                    source,
                    status,
                    start_time,
                    end_time,
                    completed_at
                )
                VALUES (%s, %s, %s, %s, 'manual', %s, %s, %s, %s)
            """

            cursor.execute(
                sql,
                (
                    user_id,
                    owner["username"],
                    location,
                    description or None,
                    status,
                    start_time,
                    end_time,
                    end_time if status == "已完成" else None
                )
            )
            workorder_id = cursor.lastrowid

        conn.commit()

    except Exception as e:
        conn.rollback()
        return f"新建工单失败：{e}", 500

    finally:
        conn.close()

    return redirect(url_for("workorder_page", workorder_id=workorder_id))


@app.route("/delete_selected_workorders", methods=["POST"])
@admin_required
def delete_selected_workorders():
    """
    批量删除选中的工单。
    删除顺序：
    1. 查询工单下的图片路径
    2. 删除 crack_results
    3. 删除 images
    4. 删除 workorders
    5. 删除 uploads 中对应原始图和检测图文件
    """

    workorder_ids = request.form.getlist("workorder_ids")

    if not workorder_ids:
        return "请先选择要删除的工单", 400

    conn = get_connection()

    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            placeholders = ",".join(["%s"] * len(workorder_ids))

            # 1. 查询这些工单下的图片ID、检测图路径、原始图路径
            select_images_sql = f"""
                SELECT id, image_path, original_image_path
                FROM images
                WHERE workorder_id IN ({placeholders})
            """
            cursor.execute(select_images_sql, workorder_ids)
            images = cursor.fetchall()

            image_ids = [img["id"] for img in images]

            image_paths = []
            for img in images:
                if img.get("image_path"):
                    image_paths.append(img["image_path"])
                if img.get("original_image_path"):
                    image_paths.append(img["original_image_path"])

            # 2. 删除 crack_results
            if image_ids:
                image_placeholders = ",".join(["%s"] * len(image_ids))
                delete_results_sql = f"""
                    DELETE FROM crack_results
                    WHERE image_id IN ({image_placeholders})
                """
                cursor.execute(delete_results_sql, image_ids)

            # 3. 删除 images
            delete_images_sql = f"""
                DELETE FROM images
                WHERE workorder_id IN ({placeholders})
            """
            cursor.execute(delete_images_sql, workorder_ids)

            # 4. 删除 workorders
            delete_workorders_sql = f"""
                DELETE FROM workorders
                WHERE id IN ({placeholders})
            """
            cursor.execute(delete_workorders_sql, workorder_ids)

        conn.commit()

        # 5. 删除本地图片文件
        upload_root = os.path.realpath(UPLOAD_FOLDER)
        for image_path in image_paths:
            if image_path:
                file_path = os.path.realpath(image_path)
                if os.path.commonpath([upload_root, file_path]) == upload_root:
                    remove_files([file_path])

    except Exception as e:
        conn.rollback()
        return f"删除工单失败：{e}", 500

    finally:
        conn.close()

    return redirect(url_for("workorders_page"))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )
