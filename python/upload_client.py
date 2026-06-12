import json
import os
import urllib.error
import urllib.request


DEFAULT_BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://192.168.73.181:5000/upload_result"
)


SEVERITY_MAP = {
    "NONE": "无",
    "MINOR": "轻微",
    "LOW": "轻微",
    "MEDIUM": "中等",
    "HIGH": "严重",
    "CRITICAL": "严重",
    "无": "无",
    "轻微": "轻微",
    "中等": "中等",
    "严重": "严重",
}


def map_severity(severity):
    return SEVERITY_MAP.get(severity, "无")


def _load_result_json(capture):
    raw = capture.get("result_json")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _image_basename(path):
    return os.path.basename(path) if path else ""


def build_capture_item(capture):
    stored = _load_result_json(capture)
    crack_count = int(capture.get("crack_count", 0) or 0)

    if crack_count > 0 and stored:
        result_json = {
            "model": stored.get("model", "YOLOv11-RKNN"),
            "bbox": stored.get("bbox", []),
            "area": int(stored.get("area", capture.get("crack_area", 0) or 0)),
            "length": float(stored.get("length", capture.get("total_length", 0) or 0)),
            "width": float(stored.get("width", capture.get("max_width", 0) or 0)),
            "severity": stored.get("severity", capture.get("severity", "NONE")),
        }
    else:
        result_json = {}

    original_filename = _image_basename(capture.get("image_path"))
    result_filename = _image_basename(capture.get("result_path"))

    return {
        "capture_id": capture.get("id"),
        "captured_at": capture.get("captured_at", ""),
        "crack_count": crack_count,
        "crack_area": int(capture.get("crack_area", 0) or 0),
        "max_width": float(capture.get("max_width", 0) or 0),
        "total_length": int(capture.get("total_length", 0) or 0),
        "severity": map_severity(capture.get("severity", "无")),
        "suggestion": capture.get("suggestion") or "No crack detected",
        "confidence": float(capture.get("confidence", 0) or 0),
        "original_image_filename": original_filename,
        "image_filename": result_filename or original_filename,
        "result_json": result_json,
    }


def collect_upload_files(pending):
    """收集待上传文件：每条 capture 包含原图 + 检测结果图。"""
    files = {}
    missing = []

    for capture in pending:
        paths = [
            ("原图", capture.get("image_path")),
            ("结果图", capture.get("result_path")),
        ]
        for label, image_path in paths:
            if not image_path:
                missing.append(f"#{capture.get('id')}: 缺少{label}路径")
                continue
            if not os.path.exists(image_path):
                missing.append(f"#{capture.get('id')}: {label}不存在 {image_path}")
                continue
            filename = os.path.basename(image_path)
            if filename in files:
                continue
            with open(image_path, "rb") as f:
                files[filename] = (filename, f.read(), "image/jpeg")

    return files, missing


def build_inspection_payload(session, captures):
    pending = [c for c in captures if not c.get("uploaded")]
    capture_items = [build_capture_item(c) for c in pending]
    return {
        "session_id": session.get("id"),
        "inspector": session.get("inspector", ""),
        "location": session.get("location", ""),
        "inspection_start_time": session.get("start_time", ""),
        "inspection_end_time": session.get("end_time") or "",
        "capture_count": len(capture_items),
        "captures": capture_items,
    }


def get_upload_url(backend_url=None):
    url = backend_url or DEFAULT_BACKEND_URL
    if url.endswith("/upload_inspection"):
        return url.replace("/upload_inspection", "/upload_result")
    if not url.endswith("/upload_result"):
        return url.rstrip("/") + "/upload_result"
    return url


def _encode_multipart(fields, files):
    boundary = "----CrackInspectionBoundary7MA4YWxkTrZu0gW"
    body = []

    for name, value in fields.items():
        body.append(f"--{boundary}\r\n".encode())
        body.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.append(value.encode("utf-8"))
        body.append(b"\r\n")

    for name, (filename, content, content_type) in files.items():
        body.append(f"--{boundary}\r\n".encode())
        body.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.append(content)
        body.append(b"\r\n")

    body.append(f"--{boundary}--\r\n".encode())
    return boundary, b"".join(body)


def _post_multipart(url, fields, files, timeout=60):
    boundary, body = _encode_multipart(fields, files)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
            status_code = response.getcode()
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="ignore")
        try:
            err_body = json.loads(err_text)
            return False, err_body.get("message", err_text)
        except json.JSONDecodeError:
            return False, f"HTTP {e.code}: {err_text}"
    except Exception as e:
        return False, f"上传失败: {e}"

    if status_code != 200:
        return False, f"HTTP {status_code}: {response_text}"

    try:
        resp = json.loads(response_text)
    except json.JSONDecodeError:
        return False, response_text

    if resp.get("code") != 200:
        return False, resp.get("message", response_text)

    return True, resp


def upload_inspection_session(session, captures, backend_url=None):
    """
    巡检结束后批量上传到 POST /upload_result（backend_new 批量接口）
    multipart 图片字段名须与 captures[].original_image_filename、image_filename 一致
    """
    backend_url = get_upload_url(backend_url)
    pending = [c for c in captures if not c.get("uploaded")]
    if not pending:
        return 0, ["没有待上传的记录"], "没有待上传的记录", [], None

    payload = build_inspection_payload(session, captures)
    print(f"[Upload] POST {backend_url}")
    print(f"[Upload] session_id={payload['session_id']} captures={payload['capture_count']}")

    files, missing = collect_upload_files(pending)
    if missing:
        return 0, missing, "图片缺失", [], None

    for item in payload["captures"]:
        for key in ("original_image_filename", "image_filename"):
            filename = item.get(key)
            if filename and filename not in files:
                return 0, [f"#{item.get('capture_id')}: JSON 引用文件未收集 {filename}"], "图片缺失", [], None

    ok, result = _post_multipart(
        backend_url,
        {"data": json.dumps(payload, ensure_ascii=False)},
        files,
        timeout=120,
    )

    if not ok:
        return 0, [str(result)], str(result), [], None

    saved_count = result.get("saved_count", len(pending))
    uploaded_ids = [c["id"] for c in pending[:saved_count]]
    summary = (
        f"上传成功: {result.get('message', 'OK')} | "
        f"saved={saved_count} image_ids={result.get('image_ids', [])}"
    )
    return saved_count, [summary], summary, uploaded_ids, result
