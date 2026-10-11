"""
Cloud Image Studio - a multi-user cloud image management system

DCC mini project : cloud-deployed web system (auth, roles, albums, batch jobs,
                   dashboard, admin panel, REST API, health check)
DAA mini project : algorithms written from scratch
                   - Bubble / Insertion / Merge / Quick sort  (gallery sorting)
                   - Linear / Binary search                   (gallery search)
                   - Seam carving (Dynamic Programming)       (content-aware resize)
                   - Huffman coding (Greedy)                  (compression analysis)
"""
from flask import (Flask, request, render_template_string, redirect, url_for, Response,
                   abort, session, g, jsonify)
from markupsafe import Markup
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import wraps
from itertools import chain
import base64
import heapq
import io
import math
import os
import random
import re
import secrets
import sqlite3
import time

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB per request
app.secret_key = os.environ.get("SECRET_KEY", "dcc-daa-mini-project-change-me")

DB_PATH = os.environ.get(
    "DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio.db")
)

# Processing steps. They are always applied in the order listed here (a pipeline).
STEPS = {
    "grayscale": "Grayscale",
    "sepia": "Sepia",
    "brighten": "Brighten",
    "darken": "Darken",
    "contrast": "High contrast",
    "sharpen": "Sharpen",
    "blur": "Blur",
    "edges": "Edge detect",
    "emboss": "Emboss",
    "invert": "Invert colors",
    "mirror": "Mirror",
    "rotate": "Rotate 90",
    "watermark": "Watermark",
}
SIZES = [200, 300, 500, 800]
PER_PAGE = 12
MAX_BATCH = 8


# =====================================================================
# DAA PART 1 : sorting and searching (no built-in sort in the first four)
# Every sort returns (sorted_list, number_of_comparisons)
# =====================================================================
def bubble_sort(arr, key):
    a = list(arr)
    comps = 0
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            comps += 1
            if key(a[j]) > key(a[j + 1]):
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:  # already sorted -> best case O(n)
            break
    return a, comps


def insertion_sort(arr, key):
    a = list(arr)
    comps = 0
    for i in range(1, len(a)):
        cur = a[i]
        cur_key = key(cur)
        j = i - 1
        while j >= 0:
            comps += 1
            if key(a[j]) > cur_key:
                a[j + 1] = a[j]
                j -= 1
            else:
                break
        a[j + 1] = cur
    return a, comps


def merge_sort(arr, key):
    comps = [0]

    def sort(a):
        if len(a) <= 1:
            return a
        mid = len(a) // 2
        left = sort(a[:mid])
        right = sort(a[mid:])
        out = []
        i = j = 0
        while i < len(left) and j < len(right):
            comps[0] += 1
            if key(left[i]) <= key(right[j]):
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    return sort(list(arr)), comps[0]


def quick_sort(arr, key):
    """Randomised quick sort (3-way partition). Each element-vs-pivot test counts as 1 comparison."""
    comps = [0]

    def sort(a):
        if len(a) <= 1:
            return a
        pivot = key(a[random.randrange(len(a))])
        less, equal, greater = [], [], []
        for x in a:
            k = key(x)
            if k < pivot:
                less.append(x)
            elif k > pivot:
                greater.append(x)
            else:
                equal.append(x)
        comps[0] += len(a)
        return sort(less) + equal + sort(greater)

    return sort(list(arr)), comps[0]


def builtin_sort(arr, key):
    return sorted(arr, key=key), None  # Python's Timsort (comparisons not counted)


ALGOS = {
    "bubble": {"name": "Bubble Sort", "fn": bubble_sort, "best": "O(n)", "avg": "O(n^2)",
               "worst": "O(n^2)", "space": "O(1)", "paradigm": "Brute force", "color": "#dc2626"},
    "insertion": {"name": "Insertion Sort", "fn": insertion_sort, "best": "O(n)", "avg": "O(n^2)",
                  "worst": "O(n^2)", "space": "O(1)", "paradigm": "Decrease and conquer", "color": "#d97706"},
    "merge": {"name": "Merge Sort", "fn": merge_sort, "best": "O(n log n)", "avg": "O(n log n)",
              "worst": "O(n log n)", "space": "O(n)", "paradigm": "Divide and conquer", "color": "#16a34a"},
    "quick": {"name": "Quick Sort (random pivot)", "fn": quick_sort, "best": "O(n log n)", "avg": "O(n log n)",
              "worst": "O(n^2) (very rare)", "space": "O(n) in this version", "paradigm": "Divide and conquer",
              "color": "#1d4ed8"},
    "builtin": {"name": "Python built-in (Timsort)", "fn": builtin_sort, "best": "O(n)", "avg": "O(n log n)",
                "worst": "O(n log n)", "space": "O(n)", "paradigm": "Hybrid merge + insertion", "color": "#7c3aed"},
}

SORT_FIELDS = {"created": "Date added", "title": "Title", "size": "File size"}

PROJECT_ALGOS = [
    ("Gallery sorting", "Bubble, Insertion, Merge, Quick sort", "Brute force / Decrease and conquer / Divide and conquer", "O(n^2) to O(n log n)"),
    ("Gallery search", "Linear search and Binary search", "Brute force / Decrease and conquer", "O(n) and O(log n)"),
    ("Content-aware resize", "Seam carving", "Dynamic programming", "O(k x W x H) for k seams"),
    ("Compression analysis", "Huffman coding", "Greedy (min-heap)", "O(N + m log m), m = distinct symbols"),
]


def sort_key(field):
    if field == "title":
        return lambda it: it["title"].lower()
    if field == "size":
        return lambda it: it["file_size"]
    return lambda it: (it["created"], it["id"])


def linear_prefix_search(items, prefix):
    """O(n): check every item."""
    comps = 0
    found = []
    for it in items:
        comps += 1
        if it["title"].lower().startswith(prefix):
            found.append(it)
    return found, comps


def binary_prefix_search(sorted_items, prefix):
    """O(log n): items MUST already be sorted by lower-case title."""
    comps = 0
    lo, hi = 0, len(sorted_items)
    while lo < hi:  # find first index whose title >= prefix
        mid = (lo + hi) // 2
        comps += 1
        if sorted_items[mid]["title"].lower() < prefix:
            lo = mid + 1
        else:
            hi = mid
    found = []
    i = lo
    while i < len(sorted_items):
        comps += 1
        if sorted_items[i]["title"].lower().startswith(prefix):
            found.append(sorted_items[i])
            i += 1
        else:
            break
    return found, comps


# =====================================================================
# DAA PART 2 : Seam carving (Dynamic Programming)
# Removes the lowest-energy vertical path of pixels again and again, so the
# image gets narrower while important content is kept.
# =====================================================================
def seam_carve(img, num):
    """img: RGB image, num: number of seams to remove. Returns (result, marked_preview, seams_removed)."""
    w, h = img.size
    num = max(0, min(num, w - 2))
    raw = img.tobytes()
    rows = [[(raw[(y * w + x) * 3], raw[(y * w + x) * 3 + 1], raw[(y * w + x) * 3 + 2])
             for x in range(w)] for y in range(h)]
    graw = ImageOps.grayscale(img).tobytes()
    gray = [list(graw[y * w:(y + 1) * w]) for y in range(h)]
    idx = [list(range(w)) for _ in range(h)]  # remembers original x of every pixel
    marked = []

    for _ in range(num):
        cw = len(rows[0])
        # 1) energy = brightness change around each pixel
        energy = []
        for y in range(h):
            g_mid = gray[y]
            g_up = gray[y - 1] if y > 0 else gray[y]
            g_dn = gray[y + 1] if y < h - 1 else gray[y]
            e_row = []
            for x in range(cw):
                left = g_mid[x - 1] if x > 0 else g_mid[x]
                right = g_mid[x + 1] if x < cw - 1 else g_mid[x]
                e_row.append(abs(right - left) + abs(g_dn[x] - g_up[x]))
            energy.append(e_row)
        # 2) dynamic programming: cheapest path from the top to every pixel
        cost = [energy[0][:]]
        for y in range(1, h):
            prev = cost[y - 1]
            cur = []
            for x in range(cw):
                best = prev[x]
                if x > 0 and prev[x - 1] < best:
                    best = prev[x - 1]
                if x < cw - 1 and prev[x + 1] < best:
                    best = prev[x + 1]
                cur.append(energy[y][x] + best)
            cost.append(cur)
        # 3) backtrack from the cheapest pixel in the last row
        last = cost[h - 1]
        x = last.index(min(last))
        seam = [0] * h
        seam[h - 1] = x
        for y in range(h - 2, -1, -1):
            lo = max(0, x - 1)
            hi = min(cw - 1, x + 1)
            row = cost[y]
            x = min(range(lo, hi + 1), key=lambda i: row[i])
            seam[y] = x
        # 4) remove the seam
        for y in range(h):
            sx = seam[y]
            marked.append((y, idx[y][sx]))
            del rows[y][sx]
            del gray[y][sx]
            del idx[y][sx]

    out_w = w - num
    result = Image.frombytes("RGB", (out_w, h), bytes(chain.from_iterable(chain.from_iterable(rows))))
    buf = bytearray(raw)
    for y, x in marked:
        i = (y * w + x) * 3
        buf[i], buf[i + 1], buf[i + 2] = 255, 0, 0
    preview = Image.frombytes("RGB", (w, h), bytes(buf))
    return result, preview, num


# =====================================================================
# DAA PART 3 : Huffman coding (Greedy algorithm with a min-heap)
# =====================================================================
def huffman_codes(freq):
    """freq: {symbol: count}. Returns {symbol: bit-string}."""
    heap = [(f, i, sym) for i, (sym, f) in enumerate(sorted(freq.items()))]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0][2]: "0"}
    counter = len(heap)
    while len(heap) > 1:  # greedy: always merge the two rarest nodes
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, counter, (n1, n2)))
        counter += 1
    codes = {}
    stack = [(heap[0][2], "")]
    while stack:
        node, code = stack.pop()
        if isinstance(node, tuple):
            stack.append((node[0], code + "0"))
            stack.append((node[1], code + "1"))
        else:
            codes[node] = code
    return codes


def huffman_report(img):
    gray = ImageOps.grayscale(img)
    data = gray.tobytes()
    freq = Counter(data)
    n = len(data)
    codes = huffman_codes(freq)
    coded_bits = sum(freq[s] * len(codes[s]) for s in freq)
    entropy = -sum((f / n) * math.log2(f / n) for f in freq.values())
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:10]
    return {
        "pixels": n,
        "symbols": len(freq),
        "fixed_bits": n * 8,
        "huff_bits": coded_bits,
        "ratio": round(100.0 * coded_bits / (n * 8), 2),
        "saved": round(100.0 - 100.0 * coded_bits / (n * 8), 2),
        "avg_len": round(coded_bits / n, 3),
        "entropy": round(entropy, 3),
        "top": [(s, f, codes[s]) for s, f in top],
    }


# =====================================================================
# DATABASE (SQLite). Images are stored as BLOBs, so no file system is needed.
# =====================================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_all(sql, params=()):
    with closing(get_db()) as db:
        return [dict(r) for r in db.execute(sql, params).fetchall()]


def db_one(sql, params=()):
    rows = db_all(sql, params)
    return rows[0] if rows else None


def db_run(sql, params=()):
    with closing(get_db()) as db:
        cur = db.execute(sql, params)
        db.commit()
        return cur.lastrowid


def init_db():
    with closing(get_db()) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created TEXT
            );
            CREATE TABLE IF NOT EXISTS albums (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created TEXT,
                UNIQUE(user_id, name)
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                album_id INTEGER REFERENCES albums(id) ON DELETE SET NULL,
                title TEXT NOT NULL,
                description TEXT,
                tags TEXT DEFAULT '',
                pipeline TEXT DEFAULT '',
                size INTEGER,
                favorite INTEGER DEFAULT 0,
                created TEXT,
                orig_w INTEGER,
                orig_h INTEGER,
                proc_w INTEGER,
                proc_h INTEGER,
                file_size INTEGER,
                original BLOB,
                processed BLOB,
                thumb BLOB
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                kind TEXT,
                total INTEGER,
                ok INTEGER,
                failed INTEGER,
                pipeline TEXT,
                duration_ms REAL,
                detail TEXT,
                created TEXT
            );
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT,
                detail TEXT,
                created TEXT
            );
            """
        )
        db.commit()


init_db()

IMG_COLS = ("i.id, i.user_id, i.album_id, i.title, i.description, i.tags, i.pipeline, i.size, i.favorite, "
            "i.created, i.orig_w, i.orig_h, i.proc_w, i.proc_h, i.file_size, a.name AS album_name")


def now_text():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_activity(action, detail="", user=None):
    user = user or g.get("user")
    db_run("INSERT INTO activity (user_id, username, action, detail, created) VALUES (?,?,?,?,?)",
           (user["id"] if user else None, user["username"] if user else "anonymous", action, detail, now_text()))


def user_images(uid):
    return db_all("SELECT " + IMG_COLS + " FROM images i LEFT JOIN albums a ON a.id = i.album_id "
                  "WHERE i.user_id = ?", (uid,))


def get_image(item_id):
    """Returns the image metadata only if it belongs to the logged-in user."""
    return db_one("SELECT " + IMG_COLS + " FROM images i LEFT JOIN albums a ON a.id = i.album_id "
                  "WHERE i.id = ? AND i.user_id = ?", (item_id, g.user["id"]))


# =====================================================================
# DEMO ACCOUNTS (created automatically on every start, handy because the
# free Render plan wipes the database when the app restarts).
# Set the environment variable DEMO_ACCOUNTS=0 to switch this off.
# =====================================================================
DEMO_ENABLED = os.environ.get("DEMO_ACCOUNTS", "1") == "1"
DEMO_USERS = [("admin", "admin123", "admin"), ("testuser", "user123", "user")]


def create_demo_accounts():
    for username, password, role in DEMO_USERS:
        if db_one("SELECT id FROM users WHERE username = ?", (username,)) is None:
            db_run("INSERT INTO users (username, password_hash, role, created) VALUES (?,?,?,?)",
                   (username, generate_password_hash(password), role, now_text()))


if DEMO_ENABLED:
    create_demo_accounts()


# =====================================================================
# IMAGE HELPERS
# =====================================================================
def apply_step(img, step):
    if step == "grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    if step == "sepia":
        return ImageOps.colorize(ImageOps.grayscale(img), black="#2b1a0a", white="#f5deb3", mid="#a67c52").convert("RGB")
    if step == "brighten":
        return ImageEnhance.Brightness(img).enhance(1.3)
    if step == "darken":
        return ImageEnhance.Brightness(img).enhance(0.7)
    if step == "contrast":
        return ImageEnhance.Contrast(img).enhance(1.5)
    if step == "sharpen":
        return img.filter(ImageFilter.SHARPEN)
    if step == "blur":
        return img.filter(ImageFilter.GaussianBlur(3))
    if step == "edges":
        return img.filter(ImageFilter.FIND_EDGES)
    if step == "emboss":
        return img.filter(ImageFilter.EMBOSS)
    if step == "invert":
        return ImageOps.invert(img)
    if step == "mirror":
        return ImageOps.mirror(img)
    if step == "rotate":
        return img.rotate(-90, expand=True)
    if step == "watermark":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        text = "Cloud Image Studio"
        pos = (10, max(0, out.size[1] - 22))
        draw.text((pos[0] + 1, pos[1] + 1), text, fill=(0, 0, 0))
        draw.text(pos, text, fill=(255, 255, 255))
        return out
    return img


def apply_pipeline(img, steps):
    out = img.convert("RGB")
    for key in STEPS:  # fixed order
        if key in steps:
            out = apply_step(out, key).convert("RGB")
    return out


def pipeline_label(p):
    names = [STEPS[s] for s in (p or "").split(",") if s in STEPS]
    return " + ".join(names) if names else "No effect (resize only)"


def to_bytes(img, fmt):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def build_blobs(img, steps, size):
    processed = apply_pipeline(img, steps)
    processed.thumbnail((size, size))
    thumb = processed.copy()
    thumb.thumbnail((220, 220))
    return to_bytes(processed, "PNG"), to_bytes(thumb, "JPEG"), processed.size


def load_upload(file_storage):
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    size = img.size
    img.thumbnail((1000, 1000))
    return img, size


def create_image(uid, title, desc, tags, album_id, steps, size, img, orig_size):
    original = img.copy()
    original.thumbnail((1000, 1000))
    processed_b, thumb_b, (pw, ph) = build_blobs(original, steps, size)
    return db_run(
        """INSERT INTO images (user_id, album_id, title, description, tags, pipeline, size, favorite, created,
                               orig_w, orig_h, proc_w, proc_h, file_size, original, processed, thumb)
           VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?,?,?,?,?)""",
        (uid, album_id, title, desc, tags, ",".join(steps), size, now_text(), orig_size[0], orig_size[1],
         pw, ph, len(processed_b), to_bytes(original, "JPEG"), processed_b, thumb_b),
    )


def clean_tags(text):
    seen = []
    for part in (text or "").split(","):
        t = part.strip().lower()
        if re.fullmatch(r"[a-z0-9 _-]{1,20}", t) and t not in seen:
            seen.append(t)
    return ",".join(seen[:5])


def read_form(uid):
    steps_in = request.form.getlist("steps")
    try:
        size = int(request.form.get("size", 300))
    except ValueError:
        size = 300
    if size not in SIZES:
        size = 300
    album_id = None
    raw_album = request.form.get("album_id", "")
    if raw_album.isdigit():
        row = db_one("SELECT id FROM albums WHERE id = ? AND user_id = ?", (int(raw_album), uid))
        if row:
            album_id = row["id"]
    return {
        "title": request.form.get("title", "").strip()[:60],
        "desc": request.form.get("description", "").strip()[:300],
        "tags": clean_tags(request.form.get("tags", "")),
        "steps": [k for k in STEPS if k in steps_in],
        "size": size,
        "album_id": album_id,
    }


def safe_next(default="/"):
    nxt = request.form.get("next", "")
    if nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return default


# =====================================================================
# SECURITY : sessions, CSRF protection, roles
# =====================================================================
def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


app.jinja_env.globals.update(csrf_token=csrf_token, plabel=pipeline_label)


@app.before_request
def before():
    g.user = None
    uid = session.get("uid")
    if uid:
        g.user = db_one("SELECT id, username, role FROM users WHERE id = ?", (uid,))
        if g.user is None:
            session.pop("uid", None)
    if request.method == "POST":
        token = session.get("csrf")
        if not token or token != request.form.get("csrf_token"):
            abort(400)


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login", msg="Please log in first."))
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if g.user["role"] != "admin":
            abort(403)
        return fn(*args, **kwargs)
    return wrapper


# =====================================================================
# PAGE TEMPLATES
# =====================================================================
BASE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} - Cloud Image Studio</title>
<style>
  :root { --bg:#eef2f7; --card:#fff; --ink:#14213d; --muted:#5b6b86; --accent:#1d4ed8;
          --accent-dark:#1e3a8a; --line:#d6deeb; --danger:#b91c1c; --ok:#166534; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--ink); line-height:1.5; }
  header { background:var(--ink); color:#fff; padding:18px 20px 0; }
  .head-in { max-width:1100px; margin:0 auto; }
  .top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; }
  header h1 { margin:0; font-size:1.5rem; }
  .who { color:#c3cde0; font-size:.9rem; display:flex; align-items:center; gap:10px; }
  .navbtn { background:transparent; color:#fff; border:1px solid #6b7da3; border-radius:6px; padding:4px 12px; cursor:pointer; font-size:.85rem; }
  .navbtn:hover { background:#24365f; }
  nav { display:flex; flex-wrap:wrap; gap:2px; margin-top:12px; }
  nav a { color:#c3cde0; text-decoration:none; padding:9px 14px; border-radius:8px 8px 0 0; font-weight:600; font-size:.93rem; }
  nav a:hover { color:#fff; }
  nav a.active { background:var(--bg); color:var(--ink); }
  main { max-width:1100px; margin:0 auto; padding:24px 16px 40px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:22px;
          box-shadow:0 6px 24px rgba(20,33,61,.07); margin-bottom:22px; }
  h2 { margin-top:0; }
  h3 { margin-bottom:8px; }
  .msg { background:#e7f6ec; color:var(--ok); border:1px solid #b7e1c4; padding:10px 14px; border-radius:8px; margin-bottom:18px; }
  .error { background:#fdecec; color:#9b1c1c; border:1px solid #f5b5b5; padding:10px 14px; border-radius:8px; margin-bottom:16px; }
  .toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; }
  label { display:flex; flex-direction:column; font-size:.88rem; color:var(--muted); flex:1; min-width:150px; }
  label.check { flex-direction:row; align-items:center; gap:8px; min-width:0; flex:none; color:var(--ink); font-size:.93rem; }
  input[type=text], input[type=password], textarea, select { margin-top:5px; padding:10px; font-size:1rem;
          border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); font-family:inherit; }
  textarea { min-height:80px; resize:vertical; }
  .btn { display:inline-block; padding:10px 20px; font-size:.95rem; font-weight:600; color:#fff; background:var(--accent);
         border:none; border-radius:8px; cursor:pointer; text-decoration:none; }
  .btn:hover { background:var(--accent-dark); }
  .btn.light { background:#e6ecf8; color:var(--accent-dark); }
  .btn.light:hover { background:#d5def2; }
  .btn.danger { background:var(--danger); }
  .btn.danger:hover { background:#7f1d1d; }
  .btn.small { padding:6px 12px; font-size:.85rem; }
  .btn:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, .drop:focus-visible, .navbtn:focus-visible
        { outline:3px solid #93b4ff; outline-offset:2px; }
  .stats { background:#f3f6fc; border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-top:16px; font-size:.92rem; }
  .stats div { margin:2px 0; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:18px; margin-bottom:22px; }
  .item { background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; display:flex; flex-direction:column; }
  .item img { width:100%; height:160px; object-fit:cover; background:#f1f4fa; display:block; }
  .item .body { padding:12px 14px; flex:1; }
  .item h3 { margin:0 0 4px; font-size:1.02rem; word-break:break-word; }
  .meta { color:var(--muted); font-size:.85rem; }
  .item .actions { display:flex; gap:6px; padding:0 14px 14px; flex-wrap:wrap; align-items:center; }
  .tag { display:inline-block; background:#e6ecf8; color:var(--accent-dark); border-radius:20px; padding:1px 10px; font-size:.78rem; font-weight:600; text-decoration:none; margin:2px 2px 2px 0; }
  .tag.album { background:#e7f6ec; color:var(--ok); }
  .empty { text-align:center; padding:30px 10px; color:var(--muted); }
  .drop { border:2px dashed #9db0d3; border-radius:12px; padding:26px 16px; text-align:center; cursor:pointer; margin-top:6px; }
  .drop:hover, .drop.over { background:#f1f5ff; border-color:var(--accent); }
  .hidden-input { display:none; }
  .fileName { margin-top:8px; font-weight:600; color:var(--accent-dark); }
  .row { display:flex; flex-wrap:wrap; gap:16px; margin-top:16px; align-items:flex-end; }
  .steps { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px 14px; margin-top:8px; }
  .compare { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; text-align:center; }
  .compare img { max-width:100%; border-radius:10px; border:1px solid var(--line); background:#f6f8fc; }
  .pix { image-rendering:pixelated; }
  table { width:100%; border-collapse:collapse; font-size:.92rem; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { background:#f3f6fc; }
  .scroll { overflow-x:auto; }
  .legend { display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; font-size:.88rem; }
  .legend span::before { content:""; display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; background:var(--c); vertical-align:middle; }
  .cards4 { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-bottom:22px; }
  .stat { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px; }
  .stat .num { font-size:1.9rem; font-weight:700; color:var(--accent-dark); }
  .bar { background:#e6ecf8; border-radius:6px; height:14px; overflow:hidden; }
  .bar span { display:block; height:100%; background:var(--accent); }
  .barrow { display:grid; grid-template-columns:130px 1fr 40px; gap:10px; align-items:center; margin:6px 0; font-size:.9rem; }
  .pager { display:flex; gap:6px; flex-wrap:wrap; justify-content:center; margin:6px 0 22px; }
  .pager a, .pager span { padding:6px 12px; border-radius:8px; background:#fff; border:1px solid var(--line); text-decoration:none; color:var(--ink); font-size:.9rem; }
  .pager span.cur { background:var(--accent); color:#fff; border-color:var(--accent); }
  .two { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:22px; }
  .auth { max-width:420px; margin:30px auto; }
  .star { background:none; border:none; font-size:1.3rem; cursor:pointer; color:#d97706; padding:0 4px; }
  footer { text-align:center; color:var(--muted); font-size:.85rem; padding-bottom:30px; }
  form.inline { display:inline; }
  code { background:#eef2f9; padding:1px 6px; border-radius:4px; }
</style>
</head>
<body>
<header>
  <div class="head-in">
    <div class="top">
      <h1>Cloud Image Studio</h1>
      {% if user %}
      <div class="who">{{ user.username }} ({{ user.role }})
        <form class="inline" method="post" action="{{ url_for('logout') }}">
          <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
          <button class="navbtn" type="submit">Log out</button>
        </form>
      </div>
      {% endif %}
    </div>
    <nav>
      {% if user %}
        <a href="{{ url_for('dashboard') }}" class="{{ 'active' if active == 'dashboard' }}">Dashboard</a>
        <a href="{{ url_for('index') }}" class="{{ 'active' if active == 'gallery' }}">Gallery</a>
        <a href="{{ url_for('new') }}" class="{{ 'active' if active == 'new' }}">Add image</a>
        <a href="{{ url_for('batch') }}" class="{{ 'active' if active == 'batch' }}">Batch upload</a>
        <a href="{{ url_for('albums') }}" class="{{ 'active' if active == 'albums' }}">Albums</a>
        <a href="{{ url_for('jobs') }}" class="{{ 'active' if active == 'jobs' }}">Jobs</a>
        <a href="{{ url_for('analysis') }}" class="{{ 'active' if active == 'analysis' }}">DAA analysis</a>
        {% if user.role == 'admin' %}<a href="{{ url_for('admin') }}" class="{{ 'active' if active == 'admin' }}">Admin</a>{% endif %}
      {% else %}
        <a href="{{ url_for('login') }}" class="{{ 'active' if active == 'login' }}">Log in</a>
        <a href="{{ url_for('register') }}" class="{{ 'active' if active == 'register' }}">Register</a>
      {% endif %}
    </nav>
  </div>
</header>
<main>
  {% if msg %}<div class="msg">{{ msg }}</div>{% endif %}
  {{ body }}
</main>
<footer>DCC + DAA Mini Project &middot; Flask, Pillow, SQLite &middot; Deployed on Render</footer>
</body>
</html>
"""

AUTH = """
<div class="card auth">
  <h2>{{ 'Create your account' if mode == 'register' else 'Log in' }}</h2>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  {% if mode == 'login' and demo %}
  <div class="stats" style="margin-top:0;margin-bottom:14px">
    <strong>Demo accounts</strong>
    <div>Admin: <code>admin</code> / <code>admin123</code></div>
    <div>Normal user: <code>testuser</code> / <code>user123</code></div>
  </div>
  {% endif %}
  <form method="post">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <div class="row"><label>Username
      <input type="text" name="username" value="{{ username }}" required maxlength="20" autocomplete="username">
    </label></div>
    <div class="row"><label>Password
      <input type="password" name="password" required minlength="6" autocomplete="{{ 'new-password' if mode == 'register' else 'current-password' }}">
    </label></div>
    {% if mode == 'register' %}
    <div class="row"><label>Confirm password
      <input type="password" name="confirm" required minlength="6" autocomplete="new-password">
    </label></div>
    <p class="meta">3 to 20 letters, numbers or underscores. The very first account created becomes the administrator.</p>
    {% endif %}
    <div class="row">
      <button class="btn" type="submit">{{ 'Create account' if mode == 'register' else 'Log in' }}</button>
      {% if mode == 'register' %}<a class="btn light" href="{{ url_for('login') }}">I have an account</a>
      {% else %}<a class="btn light" href="{{ url_for('register') }}">Create account</a>{% endif %}
    </div>
  </form>
</div>
"""

DASHBOARD = """
<div class="cards4">
  <div class="stat"><div class="num">{{ n_images }}</div>Images</div>
  <div class="stat"><div class="num">{{ storage_kb }} KB</div>Storage used</div>
  <div class="stat"><div class="num">{{ n_albums }}</div>Albums</div>
  <div class="stat"><div class="num">{{ n_fav }}</div>Favorites</div>
  <div class="stat"><div class="num">{{ n_jobs }}</div>Batch jobs</div>
</div>
<div class="two">
  <div class="card">
    <h2>Most used operations</h2>
    {% if ops %}
      {% for name, count, pct in ops %}
        <div class="barrow"><span>{{ name }}</span><div class="bar"><span style="width: {{ pct }}%"></span></div><span>{{ count }}</span></div>
      {% endfor %}
    {% else %}<p class="meta">No data yet. Add some images first.</p>{% endif %}
  </div>
  <div class="card">
    <h2>Uploads in the last 7 days</h2>
    {% for day, count, pct in days %}
      <div class="barrow"><span>{{ day }}</span><div class="bar"><span style="width: {{ pct }}%"></span></div><span>{{ count }}</span></div>
    {% endfor %}
  </div>
  <div class="card">
    <h2>Top albums</h2>
    {% if album_stats %}
      {% for a in album_stats %}
        <div class="barrow"><span>{{ a.name }}</span><div class="bar"><span style="width: {{ a.pct }}%"></span></div><span>{{ a.n }}</span></div>
      {% endfor %}
    {% else %}<p class="meta">No albums yet. <a href="{{ url_for('albums') }}">Create one</a>.</p>{% endif %}
  </div>
  <div class="card">
    <h2>Recent activity</h2>
    {% if activity %}
      <table>
        {% for a in activity %}<tr><td>{{ a.action }}<div class="meta">{{ a.detail }}</div></td><td class="meta">{{ a.created }}</td></tr>{% endfor %}
      </table>
    {% else %}<p class="meta">Nothing yet.</p>{% endif %}
  </div>
</div>
"""

GALLERY = """
<div class="card">
  <form method="get" class="toolbar">
    <label>Search title (starts with)
      <input type="text" name="q" value="{{ q }}" placeholder="e.g. sun">
    </label>
    <label>Search method
      <select name="method">
        <option value="linear" {{ 'selected' if method == 'linear' }}>Linear search - O(n)</option>
        <option value="binary" {{ 'selected' if method == 'binary' }}>Binary search - O(log n)</option>
      </select>
    </label>
    <label>Album
      <select name="album">
        <option value="">All albums</option>
        <option value="none" {{ 'selected' if album == 'none' }}>No album</option>
        {% for a in album_list %}<option value="{{ a.id }}" {{ 'selected' if album == a.id|string }}>{{ a.name }}</option>{% endfor %}
      </select>
    </label>
    <label>Sort by
      <select name="sort_by">
        {% for k, v in sort_fields.items() %}<option value="{{ k }}" {{ 'selected' if k == sort_by }}>{{ v }}</option>{% endfor %}
      </select>
    </label>
    <label>Order
      <select name="order">
        <option value="asc" {{ 'selected' if order == 'asc' }}>Ascending</option>
        <option value="desc" {{ 'selected' if order == 'desc' }}>Descending</option>
      </select>
    </label>
    <label>Sorting algorithm
      <select name="algo">
        {% for k, a in algos.items() %}<option value="{{ k }}" {{ 'selected' if k == algo }}>{{ a.name }}</option>{% endfor %}
      </select>
    </label>
    <label class="check"><input type="checkbox" name="fav" value="1" {{ 'checked' if fav }}> Favorites only</label>
    <button class="btn" type="submit">Apply</button>
    <a class="btn light" href="{{ url_for('index') }}">Reset</a>
  </form>
  <div class="stats">
    <div><strong>{{ shown }}</strong> of <strong>{{ total }}</strong> images match
      {% if tag %}(tag: <strong>{{ tag }}</strong>){% endif %} &middot; page {{ page_no }} of {{ pages }}</div>
    <div>Sort: {{ algos[algo].name }} (average {{ algos[algo].avg }}) &mdash;
      {% if sort_comps is none %}comparisons not counted{% else %}{{ sort_comps }} comparisons{% endif %},
      {{ sort_ms }} ms</div>
    {% if q %}
      <div>Search: {{ 'Binary' if method == 'binary' else 'Linear' }} for "{{ q }}" &mdash; {{ search_comps }} comparisons
        {% if method == 'binary' %}(plus {{ prep_comps }} comparisons to sort by title first using Merge Sort){% endif %}</div>
    {% endif %}
  </div>
</div>

{% if page_items %}
<div class="grid">
  {% for it in page_items %}
  <div class="item">
    <a href="{{ url_for('view', item_id=it.id) }}"><img src="{{ url_for('image', item_id=it.id, kind='thumb') }}" alt="{{ it.title }}"></a>
    <div class="body">
      <h3>{{ it.title }}</h3>
      {% if it.album_name %}<span class="tag album">{{ it.album_name }}</span>{% endif %}
      {% for t in it.tags.split(',') if t %}<a class="tag" href="{{ url_for('index', tag=t) }}">{{ t }}</a>{% endfor %}
      <div class="meta">{{ plabel(it.pipeline) }}</div>
      <div class="meta">{{ (it.file_size / 1024) | round(1) }} KB &middot; {{ it.created }}</div>
    </div>
    <div class="actions">
      <a class="btn small" href="{{ url_for('view', item_id=it.id) }}">View</a>
      <a class="btn small light" href="{{ url_for('edit', item_id=it.id) }}">Edit</a>
      <form class="inline" method="post" action="{{ url_for('favorite', item_id=it.id) }}">
        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
        <input type="hidden" name="next" value="{{ request.full_path }}">
        <button class="star" type="submit" title="Toggle favorite">{{ '&#9733;'|safe if it.favorite else '&#9734;'|safe }}</button>
      </form>
      <form class="inline" method="post" action="{{ url_for('delete', item_id=it.id) }}"
            onsubmit="return confirm('Delete this image permanently?');">
        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
        <button class="btn small danger" type="submit">Delete</button>
      </form>
    </div>
  </div>
  {% endfor %}
</div>
{% if pages > 1 %}
<div class="pager">
  {% if page_no > 1 %}<a href="{{ url_for('index', page=page_no - 1, **args) }}">Previous</a>{% endif %}
  {% for p in range(1, pages + 1) %}
    {% if p == page_no %}<span class="cur">{{ p }}</span>{% else %}<a href="{{ url_for('index', page=p, **args) }}">{{ p }}</a>{% endif %}
  {% endfor %}
  {% if page_no < pages %}<a href="{{ url_for('index', page=page_no + 1, **args) }}">Next</a>{% endif %}
</div>
{% endif %}
{% else %}
<div class="card empty">
  {% if total == 0 %}
    <h2>Your gallery is empty</h2>
    <p>Add an image, upload several at once, or fill the gallery with sample data to try sorting and searching.</p>
    <a class="btn" href="{{ url_for('new') }}">Add image</a>
    <form class="inline" method="post" action="{{ url_for('seed') }}">
      <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
      <button class="btn light" type="submit">Add 30 sample images</button></form>
  {% else %}
    <h2>No images match these filters</h2>
    <p>Try a shorter search or press Reset.</p>
  {% endif %}
</div>
{% endif %}

{% if total_all > 0 %}
<div style="display:flex; gap:10px; flex-wrap:wrap;">
  <form method="post" action="{{ url_for('seed') }}">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <button class="btn light" type="submit">Add 30 sample images</button></form>
  <form method="post" action="{{ url_for('clear') }}" onsubmit="return confirm('Delete ALL your images?');">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <button class="btn danger" type="submit">Delete all my images</button></form>
</div>
{% endif %}
"""

STEPS_BOX = """
<div class="row"><div style="width:100%">
  <div class="meta">Processing pipeline (steps run in the order shown)</div>
  <div class="steps">
    {% for k, v in steps_map.items() %}
      <label class="check"><input type="checkbox" name="steps" value="{{ k }}" {{ 'checked' if k in steps_val }}> {{ v }}</label>
    {% endfor %}
  </div>
</div></div>
"""

DROP_JS = """
<script>
  (function () {
    var drop = document.getElementById("drop");
    var input = document.getElementById("fileInput");
    var nameBox = document.getElementById("fileName");
    function showName() {
      var names = [];
      for (var i = 0; i < input.files.length; i++) { names.push(input.files[i].name); }
      nameBox.textContent = names.length ? "Selected: " + names.join(", ") : "";
    }
    drop.addEventListener("click", function () { input.click(); });
    drop.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
    input.addEventListener("change", showName);
    drop.addEventListener("dragover", function (e) { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", function () { drop.classList.remove("over"); });
    drop.addEventListener("drop", function (e) {
      e.preventDefault(); drop.classList.remove("over");
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; showName(); }
    });
  })();
</script>
"""

FORM = """
<div class="card">
  <h2>{{ 'Edit image' if mode == 'edit' else 'Add a new image' }}</h2>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" enctype="multipart/form-data">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    {% if mode == 'new' %}
      <div class="drop" id="drop" tabindex="0">
        <strong>Click to choose an image, or drag it here</strong><br>
        <span class="meta">JPG, PNG, WEBP or BMP</span>
        <div class="fileName" id="fileName"></div>
      </div>
      <input class="hidden-input" type="file" name="image" id="fileInput" accept="image/*" required>
    {% else %}
      <p class="meta">The original image stays saved. Changing the pipeline or size re-processes it.</p>
    {% endif %}
    <div class="row">
      <label>Title
        <input type="text" name="title" value="{{ f.title }}" maxlength="60" required placeholder="e.g. Sunset at the beach">
      </label>
      <label>Album
        <select name="album_id">
          <option value="">No album</option>
          {% for a in album_list %}<option value="{{ a.id }}" {{ 'selected' if f.album_id == a.id }}>{{ a.name }}</option>{% endfor %}
        </select>
      </label>
    </div>
    <div class="row">
      <label>Description (optional)
        <textarea name="description" maxlength="300">{{ f.desc }}</textarea>
      </label>
    </div>
    <div class="row">
      <label>Tags (comma separated, up to 5)
        <input type="text" name="tags" value="{{ f.tags }}" placeholder="nature, travel">
      </label>
      <label>Maximum size
        <select name="size">
          {% for s in sizes %}<option value="{{ s }}" {{ 'selected' if s == f.size }}>{{ s }} px</option>{% endfor %}
        </select>
      </label>
    </div>
    @@STEPS@@
    <div class="row">
      <button class="btn" type="submit">{{ 'Save changes' if mode == 'edit' else 'Process and save' }}</button>
      <a class="btn light" href="{{ url_for('index') }}">Cancel</a>
    </div>
  </form>
</div>
{% if mode == 'new' %}@@JS@@{% endif %}
""".replace("@@STEPS@@", STEPS_BOX).replace("@@JS@@", DROP_JS)

BATCH = """
<div class="card">
  <h2>Batch upload</h2>
  <p class="meta">Upload up to {{ max_batch }} images at once. The same pipeline is applied to all of them and the run is recorded as a job.</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" enctype="multipart/form-data">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <div class="drop" id="drop" tabindex="0">
      <strong>Click to choose several images, or drag them here</strong><br>
      <span class="meta">The file name becomes the image title</span>
      <div class="fileName" id="fileName"></div>
    </div>
    <input class="hidden-input" type="file" name="images" id="fileInput" accept="image/*" multiple required>
    <div class="row">
      <label>Album
        <select name="album_id">
          <option value="">No album</option>
          {% for a in album_list %}<option value="{{ a.id }}" {{ 'selected' if f.album_id == a.id }}>{{ a.name }}</option>{% endfor %}
        </select>
      </label>
      <label>Tags (comma separated)
        <input type="text" name="tags" value="{{ f.tags }}" placeholder="holiday, 2026">
      </label>
      <label>Maximum size
        <select name="size">
          {% for s in sizes %}<option value="{{ s }}" {{ 'selected' if s == f.size }}>{{ s }} px</option>{% endfor %}
        </select>
      </label>
    </div>
    @@STEPS@@
    <div class="row">
      <button class="btn" type="submit">Process batch</button>
      <a class="btn light" href="{{ url_for('jobs') }}">View job history</a>
    </div>
  </form>
</div>
@@JS@@
""".replace("@@STEPS@@", STEPS_BOX).replace("@@JS@@", DROP_JS)

VIEW = """
<div class="card">
  <h2>{{ it.title }} {% if it.favorite %}<span style="color:#d97706">&#9733;</span>{% endif %}</h2>
  <p>{{ it.description or 'No description.' }}</p>
  <div>
    {% if it.album_name %}<span class="tag album">Album: {{ it.album_name }}</span>{% endif %}
    {% for t in it.tags.split(',') if t %}<a class="tag" href="{{ url_for('index', tag=t) }}">{{ t }}</a>{% endfor %}
  </div>
  <p class="meta">Pipeline: {{ plabel(it.pipeline) }}</p>
  <div class="compare">
    <div>
      <h3>Original</h3>
      <img src="{{ url_for('image', item_id=it.id, kind='original') }}" alt="Original" style="max-height:420px;">
      <div class="meta">Uploaded size: {{ it.orig_w }} x {{ it.orig_h }} px</div>
    </div>
    <div>
      <h3>Processed</h3>
      <img src="{{ url_for('image', item_id=it.id, kind='processed') }}" alt="Processed" style="max-height:420px;">
      <div class="meta">Output size: {{ it.proc_w }} x {{ it.proc_h }} px &middot; {{ (it.file_size / 1024) | round(1) }} KB</div>
    </div>
  </div>
  <div class="row">
    <a class="btn" href="{{ url_for('download', item_id=it.id) }}">Download</a>
    <a class="btn light" href="{{ url_for('edit', item_id=it.id) }}">Edit</a>
    <a class="btn light" href="{{ url_for('lab', item_id=it.id) }}">Algorithm lab</a>
    <form class="inline" method="post" action="{{ url_for('favorite', item_id=it.id) }}">
      <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
      <input type="hidden" name="next" value="{{ url_for('view', item_id=it.id) }}">
      <button class="btn light" type="submit">{{ 'Remove favorite' if it.favorite else 'Mark favorite' }}</button>
    </form>
    <form class="inline" method="post" action="{{ url_for('delete', item_id=it.id) }}"
          onsubmit="return confirm('Delete this image permanently?');">
      <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
      <button class="btn danger" type="submit">Delete</button>
    </form>
    <a class="btn light" href="{{ url_for('index') }}">Back to gallery</a>
  </div>
  <p class="meta">Added {{ it.created }} (UTC)</p>
</div>
"""

LAB = """
<div class="card">
  <h2>Algorithm lab: {{ it.title }}</h2>
  <p class="meta">Run the DAA algorithms on this image. <a href="{{ url_for('view', item_id=it.id) }}">Back to image</a></p>
</div>

<div class="card">
  <h2>Seam carving (Dynamic Programming)</h2>
  <p>Makes the image narrower by repeatedly removing the lowest-energy vertical path of pixels (a seam). Dynamic programming finds
     each seam: <code>cost[y][x] = energy[y][x] + min(cost[y-1][x-1], cost[y-1][x], cost[y-1][x+1])</code>.
     Work is done on a small copy (max 160 px wide) because the algorithm is O(k x W x H).</p>
  <form method="get" class="toolbar">
    <input type="hidden" name="seam" value="1">
    <label>Shrink the width by
      <select name="pct">
        {% for p in [10, 20, 30, 40] %}<option value="{{ p }}" {{ 'selected' if p == pct }}>{{ p }}%</option>{% endfor %}
      </select>
    </label>
    <button class="btn" type="submit">Run seam carving</button>
  </form>
  {% if seam %}
  <div class="stats">
    <div>Input: {{ seam.w }} x {{ seam.h }} px &rarr; Output: {{ seam.out_w }} x {{ seam.h }} px</div>
    <div>Seams removed: {{ seam.k }} &middot; Time: {{ seam.ms }} ms &middot; Approx. DP cell updates: {{ seam.cells }}</div>
  </div>
  <div class="compare" style="margin-top:16px">
    <div><h3>Seams found (red)</h3><img class="pix" src="data:image/png;base64,{{ seam.preview }}" alt="Seams marked in red"></div>
    <div><h3>Seam carving result</h3><img class="pix" src="data:image/png;base64,{{ seam.result }}" alt="Seam carved image"></div>
    <div><h3>Normal resize (squashed)</h3><img class="pix" src="data:image/png;base64,{{ seam.squash }}" alt="Normally resized image"></div>
  </div>
  {% endif %}
</div>

<div class="card">
  <h2>Huffman coding (Greedy)</h2>
  <p>Estimates how well the grayscale pixels of the processed image compress with Huffman coding. A min-heap repeatedly merges
     the two rarest values, so frequent pixel values get short bit codes.</p>
  <form method="get" class="toolbar">
    <input type="hidden" name="huff" value="1">
    <button class="btn" type="submit">Run Huffman analysis</button>
  </form>
  {% if huff %}
  <div class="stats">
    <div>Pixels: {{ huff.pixels }} &middot; Distinct values: {{ huff.symbols }}</div>
    <div>Fixed length (8 bits per pixel): {{ huff.fixed_bits }} bits</div>
    <div>Huffman coded: {{ huff.huff_bits }} bits ({{ huff.ratio }}% of original, {{ huff.saved }}% saved)</div>
    <div>Average code length: {{ huff.avg_len }} bits per pixel &middot; Entropy (theoretical minimum): {{ huff.entropy }} bits</div>
    <div>Time: {{ huff_ms }} ms</div>
  </div>
  <h3>Ten most frequent pixel values</h3>
  <table>
    <tr><th>Pixel value</th><th>Count</th><th>Huffman code</th></tr>
    {% for s, f, c in huff.top %}<tr><td>{{ s }}</td><td>{{ f }}</td><td><code>{{ c }}</code></td></tr>{% endfor %}
  </table>
  {% endif %}
</div>
"""

ALBUMS = """
<div class="card">
  <h2>Create an album</h2>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="post" action="{{ url_for('albums') }}" class="toolbar">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <label>Album name
      <input type="text" name="name" maxlength="30" required placeholder="e.g. Holiday 2026">
    </label>
    <button class="btn" type="submit">Create album</button>
  </form>
</div>
<div class="card">
  <h2>Your albums</h2>
  {% if album_rows %}
  <div class="scroll"><table>
    <tr><th>Name</th><th>Images</th><th>Created</th><th>Actions</th></tr>
    {% for a in album_rows %}
    <tr>
      <td><a href="{{ url_for('index', album=a.id) }}">{{ a.name }}</a></td>
      <td>{{ a.n }}</td>
      <td class="meta">{{ a.created }}</td>
      <td>
        <form class="inline" method="post" action="{{ url_for('album_rename', album_id=a.id) }}">
          <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
          <input type="text" name="name" value="{{ a.name }}" maxlength="30" required style="margin:0;padding:6px;width:150px">
          <button class="btn small light" type="submit">Rename</button>
        </form>
        <form class="inline" method="post" action="{{ url_for('album_delete', album_id=a.id) }}"
              onsubmit="return confirm('Delete this album? Its images are kept.');">
          <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
          <button class="btn small danger" type="submit">Delete</button>
        </form>
      </td>
    </tr>
    {% endfor %}
  </table></div>
  {% else %}<p class="meta">You have no albums yet.</p>{% endif %}
</div>
"""

JOBS = """
<div class="card">
  <h2>Batch job history</h2>
  {% if rows %}
  <div class="scroll"><table>
    <tr><th>#</th><th>When (UTC)</th><th>Pipeline</th><th>Files</th><th>Succeeded</th><th>Failed</th><th>Time</th><th>Notes</th></tr>
    {% for j in rows %}
    <tr><td>{{ j.id }}</td><td>{{ j.created }}</td><td>{{ plabel(j.pipeline) }}</td><td>{{ j.total }}</td>
        <td>{{ j.ok }}</td><td>{{ j.failed }}</td><td>{{ '%.0f' % j.duration_ms }} ms</td><td class="meta">{{ j.detail }}</td></tr>
    {% endfor %}
  </table></div>
  {% else %}<p class="meta">No batch jobs yet. Use <a href="{{ url_for('batch') }}">Batch upload</a>.</p>{% endif %}
</div>
"""

ADMIN = """
<div class="cards4">
  <div class="stat"><div class="num">{{ totals.users }}</div>Users</div>
  <div class="stat"><div class="num">{{ totals.images }}</div>Images</div>
  <div class="stat"><div class="num">{{ totals.storage_kb }} KB</div>Storage</div>
  <div class="stat"><div class="num">{{ totals.jobs }}</div>Jobs run</div>
</div>
<div class="card">
  <h2>User management</h2>
  <div class="scroll"><table>
    <tr><th>User</th><th>Role</th><th>Images</th><th>Storage</th><th>Joined</th><th>Actions</th></tr>
    {% for u in users %}
    <tr>
      <td>{{ u.username }}</td><td>{{ u.role }}</td><td>{{ u.n_images }}</td>
      <td>{{ (u.storage / 1024) | round(1) }} KB</td><td class="meta">{{ u.created }}</td>
      <td>
        {% if u.id != user.id %}
        <form class="inline" method="post" action="{{ url_for('admin_toggle', uid=u.id) }}">
          <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
          <button class="btn small light" type="submit">{{ 'Make user' if u.role == 'admin' else 'Make admin' }}</button>
        </form>
        <form class="inline" method="post" action="{{ url_for('admin_delete', uid=u.id) }}"
              onsubmit="return confirm('Delete this user and all their data?');">
          <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
          <button class="btn small danger" type="submit">Delete</button>
        </form>
        {% else %}<span class="meta">(you)</span>{% endif %}
      </td>
    </tr>
    {% endfor %}
  </table></div>
</div>
<div class="card">
  <h2>Activity log (latest 40)</h2>
  <div class="scroll"><table>
    <tr><th>When (UTC)</th><th>User</th><th>Action</th><th>Detail</th></tr>
    {% for a in activity %}<tr><td class="meta">{{ a.created }}</td><td>{{ a.username }}</td><td>{{ a.action }}</td><td class="meta">{{ a.detail }}</td></tr>{% endfor %}
  </table></div>
</div>
"""

ANALYSIS = """
<div class="card">
  <h2>Algorithm analysis (DAA)</h2>
  <p>Every algorithm below is written from scratch in this project. Run the benchmark to measure how running time grows with the input size n.</p>
  <form method="get" class="toolbar">
    <input type="hidden" name="run" value="1">
    <label>Input type
      <select name="case">
        <option value="random" {{ 'selected' if case == 'random' }}>Random (average case)</option>
        <option value="sorted" {{ 'selected' if case == 'sorted' }}>Already sorted (best case for some)</option>
        <option value="reversed" {{ 'selected' if case == 'reversed' }}>Reverse sorted (worst case for some)</option>
      </select>
    </label>
    <button class="btn" type="submit">Run benchmark</button>
  </form>
</div>

{% if results %}
<div class="card">
  <h2>Results: {{ case_name }}</h2>
  <div class="scroll">
  <table>
    <tr><th>Algorithm</th>{% for n in sizes %}<th>n = {{ n }}</th>{% endfor %}</tr>
    {% for k, a in algos.items() %}
    <tr><td>{{ a.name }}</td>{% for v in results[k] %}<td>{{ '%.2f' % v }} ms</td>{% endfor %}</tr>
    {% endfor %}
  </table>
  </div>
  <h3>Running time vs input size</h3>
  {{ chart }}
  <div class="legend">
    {% for k, a in algos.items() %}<span style="--c:{{ a.color }}">{{ a.name }}</span>{% endfor %}
  </div>
</div>
{% endif %}

<div class="card">
  <h2>Sorting and searching complexity</h2>
  <div class="scroll">
  <table>
    <tr><th>Algorithm</th><th>Best</th><th>Average</th><th>Worst</th><th>Space</th><th>Design technique</th></tr>
    {% for k, a in algos.items() %}
    <tr><td>{{ a.name }}</td><td>{{ a.best }}</td><td>{{ a.avg }}</td><td>{{ a.worst }}</td><td>{{ a.space }}</td><td>{{ a.paradigm }}</td></tr>
    {% endfor %}
    <tr><td>Linear search</td><td>O(1)</td><td>O(n)</td><td>O(n)</td><td>O(1)</td><td>Brute force</td></tr>
    <tr><td>Binary search</td><td>O(1)</td><td>O(log n)</td><td>O(log n)</td><td>O(1)</td><td>Decrease and conquer (needs sorted data)</td></tr>
  </table>
  </div>
</div>

<div class="card">
  <h2>Where each algorithm is used in the system</h2>
  <div class="scroll"><table>
    <tr><th>Feature</th><th>Algorithm</th><th>Design technique</th><th>Complexity</th></tr>
    {% for f, a, p, c in project_algos %}<tr><td>{{ f }}</td><td>{{ a }}</td><td>{{ p }}</td><td>{{ c }}</td></tr>{% endfor %}
  </table></div>
  <p class="meta">Sorting and searching run in the Gallery. Seam carving and Huffman coding run from the "Algorithm lab" button on any image.</p>
</div>
"""

NOTICE = """
<div class="card empty">
  <h2>{{ heading }}</h2>
  <p>{{ text }}</p>
  <a class="btn" href="{{ url_for('index') }}">Go to gallery</a>
</div>
"""


def page(body_template, title="", active="", **ctx):
    ctx.setdefault("user", g.get("user"))
    body = Markup(render_template_string(body_template, **ctx))
    return render_template_string(
        BASE, body=body, title=title, active=active, user=g.get("user"), msg=request.args.get("msg", "")
    )


# =====================================================================
# ERROR PAGES
# =====================================================================
@app.errorhandler(400)
def bad_request(e):
    return page(NOTICE, heading="Session expired",
                text="The form could not be verified. Go back, refresh the page and try again."), 400


@app.errorhandler(403)
def forbidden(e):
    return page(NOTICE, heading="Not allowed", text="You do not have permission to open this page."), 403


@app.errorhandler(404)
def not_found(e):
    return page(NOTICE, heading="Not found", text="That page or image does not exist (it may have been deleted)."), 404


@app.errorhandler(413)
def too_large(e):
    return page(NOTICE, heading="Upload too large", text="The upload is over 20 MB. Please send fewer or smaller images."), 413


# =====================================================================
# AUTHENTICATION
# =====================================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("index"))
    error = ""
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):
            error = "Username must be 3 to 20 letters, numbers or underscores."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "The two passwords do not match."
        else:
            role = "admin" if db_one("SELECT COUNT(*) AS n FROM users")["n"] == 0 else "user"
            try:
                uid = db_run("INSERT INTO users (username, password_hash, role, created) VALUES (?,?,?,?)",
                             (username, generate_password_hash(password), role, now_text()))
            except sqlite3.IntegrityError:
                error = "That username is already taken."
            else:
                user = {"id": uid, "username": username, "role": role}
                session["uid"] = uid
                log_activity("Registered", "role: " + role, user)
                return redirect(url_for("dashboard", msg="Welcome, %s! Your account is ready." % username))
    return page(AUTH, title="Register", active="register", mode="register", error=error, username=username)


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("index"))
    error = ""
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = db_one("SELECT id, username, role, password_hash FROM users WHERE username = ?", (username,))
        if row and check_password_hash(row["password_hash"], password):
            session["uid"] = row["id"]
            log_activity("Logged in", "", row)
            return redirect(url_for("dashboard"))
        error = "Wrong username or password."
    return page(AUTH, title="Log in", active="login", mode="login", error=error, username=username,
                demo=DEMO_ENABLED)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    log_activity("Logged out")
    session.pop("uid", None)
    return redirect(url_for("login", msg="You have been logged out."))


# =====================================================================
# DASHBOARD
# =====================================================================
def build_stats(uid):
    imgs = db_all("SELECT pipeline, file_size, favorite, created FROM images WHERE user_id = ?", (uid,))
    ops = Counter()
    for im in imgs:
        for s in (im["pipeline"] or "").split(","):
            if s in STEPS:
                ops[STEPS[s]] += 1
    ops_list = ops.most_common(8)
    top = ops_list[0][1] if ops_list else 1
    ops_rows = [(n, c, int(100 * c / top)) for n, c in ops_list]

    today = datetime.now(timezone.utc).date()
    per_day = Counter(im["created"][:10] for im in imgs if im["created"])
    day_names = [(today - timedelta(days=d)) for d in range(6, -1, -1)]
    max_day = max([per_day.get(str(d), 0) for d in day_names] + [1])
    day_rows = [(d.strftime("%d %b"), per_day.get(str(d), 0), int(100 * per_day.get(str(d), 0) / max_day))
                for d in day_names]

    album_stats = db_all("SELECT a.name AS name, COUNT(i.id) AS n FROM albums a LEFT JOIN images i ON i.album_id = a.id "
                         "WHERE a.user_id = ? GROUP BY a.id ORDER BY n DESC, a.name LIMIT 5", (uid,))
    top_album = max([a["n"] for a in album_stats] + [1])
    for a in album_stats:
        a["pct"] = int(100 * a["n"] / top_album)

    return {
        "n_images": len(imgs),
        "storage_kb": round(sum(i["file_size"] or 0 for i in imgs) / 1024, 1),
        "n_albums": db_one("SELECT COUNT(*) AS n FROM albums WHERE user_id = ?", (uid,))["n"],
        "n_fav": sum(1 for i in imgs if i["favorite"]),
        "n_jobs": db_one("SELECT COUNT(*) AS n FROM jobs WHERE user_id = ?", (uid,))["n"],
        "ops": ops_rows, "days": day_rows, "album_stats": album_stats,
    }


@app.route("/dashboard")
@login_required
def dashboard():
    stats = build_stats(g.user["id"])
    activity = db_all("SELECT action, detail, created FROM activity WHERE user_id = ? ORDER BY id DESC LIMIT 8",
                      (g.user["id"],))
    return page(DASHBOARD, title="Dashboard", active="dashboard", activity=activity, **stats)


# =====================================================================
# GALLERY : READ (list, filter, search, sort, paginate)
# =====================================================================
@app.route("/")
@login_required
def index():
    uid = g.user["id"]
    q = request.args.get("q", "").strip().lower()
    method = request.args.get("method", "linear")
    sort_by = request.args.get("sort_by", "created")
    order = request.args.get("order", "desc")
    algo = request.args.get("algo", "merge")
    album = request.args.get("album", "")
    fav = request.args.get("fav", "")
    tag = request.args.get("tag", "").strip().lower()
    try:
        page_no = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page_no = 1
    if method not in ("linear", "binary"):
        method = "linear"
    if sort_by not in SORT_FIELDS:
        sort_by = "created"
    if order not in ("asc", "desc"):
        order = "desc"
    if algo not in ALGOS:
        algo = "merge"
    if album != "none" and not album.isdigit():
        album = ""
    fav = "1" if fav == "1" else ""

    everything = user_images(uid)
    items = everything
    if album == "none":
        items = [i for i in items if i["album_id"] is None]
    elif album:
        items = [i for i in items if i["album_id"] == int(album)]
    if fav:
        items = [i for i in items if i["favorite"]]
    if tag:
        items = [i for i in items if tag in (i["tags"] or "").split(",")]

    search_comps = 0
    prep_comps = 0
    if q:
        if method == "binary":
            by_title, prep_comps = merge_sort(items, sort_key("title"))
            items, search_comps = binary_prefix_search(by_title, q)
        else:
            items, search_comps = linear_prefix_search(items, q)

    start = time.perf_counter()
    items, sort_comps = ALGOS[algo]["fn"](items, sort_key(sort_by))
    sort_ms = round((time.perf_counter() - start) * 1000, 3)
    if order == "desc":
        items = items[::-1]

    pages = max(1, math.ceil(len(items) / PER_PAGE))
    page_no = min(page_no, pages)
    page_items = items[(page_no - 1) * PER_PAGE: page_no * PER_PAGE]

    args = {k: v for k, v in dict(q=q, method=method, sort_by=sort_by, order=order, algo=algo,
                                  album=album, fav=fav, tag=tag).items() if v}
    album_list = db_all("SELECT id, name FROM albums WHERE user_id = ? ORDER BY name", (uid,))
    return page(
        GALLERY, title="Gallery", active="gallery",
        page_items=page_items, total=len(items), total_all=len(everything), shown=len(items),
        page_no=page_no, pages=pages, args=args, q=q, method=method, sort_by=sort_by, order=order,
        algo=algo, album=album, fav=fav, tag=tag, album_list=album_list, algos=ALGOS,
        sort_fields=SORT_FIELDS, sort_comps=sort_comps, sort_ms=sort_ms,
        search_comps=search_comps, prep_comps=prep_comps,
    )


# =====================================================================
# IMAGES : CREATE / READ / UPDATE / DELETE
# =====================================================================
def form_page(mode, error="", f=None):
    f = f or {"title": "", "desc": "", "tags": "", "steps": ["grayscale"], "size": 300, "album_id": None}
    album_list = db_all("SELECT id, name FROM albums WHERE user_id = ? ORDER BY name", (g.user["id"],))
    return page(FORM, title="Add image" if mode == "new" else "Edit image",
                active="new" if mode == "new" else "gallery", mode=mode, error=error, f=f,
                album_list=album_list, sizes=SIZES, steps_map=STEPS, steps_val=f["steps"])


@app.route("/new", methods=["GET", "POST"])
@login_required
def new():
    if request.method == "POST":
        f = read_form(g.user["id"])
        if not f["title"]:
            return form_page("new", "Please enter a title.", f)
        file = request.files.get("image")
        if file is None or file.filename == "":
            return form_page("new", "Please choose an image.", f)
        try:
            img, orig_size = load_upload(file)
        except Exception:
            return form_page("new", "Could not read that file. Upload a valid JPG, PNG, WEBP or BMP image.", f)
        new_id = create_image(g.user["id"], f["title"], f["desc"], f["tags"], f["album_id"], f["steps"],
                              f["size"], img, orig_size)
        log_activity("Added image", f["title"])
        return redirect(url_for("view", item_id=new_id, msg="Image saved."))
    return form_page("new")


@app.route("/view/<int:item_id>")
@login_required
def view(item_id):
    it = get_image(item_id)
    if it is None:
        abort(404)
    return page(VIEW, title=it["title"], active="gallery", it=it)


@app.route("/edit/<int:item_id>", methods=["GET", "POST"])
@login_required
def edit(item_id):
    it = get_image(item_id)
    if it is None:
        abort(404)
    if request.method == "POST":
        f = read_form(g.user["id"])
        if not f["title"]:
            return form_page("edit", "Please enter a title.", f)
        row = db_one("SELECT original FROM images WHERE id = ?", (item_id,))
        img = Image.open(io.BytesIO(row["original"])).convert("RGB")
        processed_b, thumb_b, (pw, ph) = build_blobs(img, f["steps"], f["size"])
        db_run("""UPDATE images SET title=?, description=?, tags=?, album_id=?, pipeline=?, size=?, proc_w=?,
                  proc_h=?, file_size=?, processed=?, thumb=? WHERE id=? AND user_id=?""",
               (f["title"], f["desc"], f["tags"], f["album_id"], ",".join(f["steps"]), f["size"], pw, ph,
                len(processed_b), processed_b, thumb_b, item_id, g.user["id"]))
        log_activity("Edited image", f["title"])
        return redirect(url_for("view", item_id=item_id, msg="Changes saved."))
    f = {"title": it["title"], "desc": it["description"] or "", "tags": it["tags"] or "",
         "steps": [s for s in (it["pipeline"] or "").split(",") if s in STEPS], "size": it["size"],
         "album_id": it["album_id"]}
    return form_page("edit", "", f)


@app.route("/delete/<int:item_id>", methods=["POST"])
@login_required
def delete(item_id):
    it = get_image(item_id)
    if it is None:
        abort(404)
    db_run("DELETE FROM images WHERE id = ? AND user_id = ?", (item_id, g.user["id"]))
    log_activity("Deleted image", it["title"])
    return redirect(url_for("index", msg="Image deleted."))


@app.route("/favorite/<int:item_id>", methods=["POST"])
@login_required
def favorite(item_id):
    it = get_image(item_id)
    if it is None:
        abort(404)
    db_run("UPDATE images SET favorite = ? WHERE id = ? AND user_id = ?",
           (0 if it["favorite"] else 1, item_id, g.user["id"]))
    return redirect(safe_next(url_for("index")))


@app.route("/clear", methods=["POST"])
@login_required
def clear():
    db_run("DELETE FROM images WHERE user_id = ?", (g.user["id"],))
    log_activity("Deleted all images")
    return redirect(url_for("index", msg="All your images were deleted."))


KINDS = {"original": ("original", "image/jpeg"), "processed": ("processed", "image/png"),
         "thumb": ("thumb", "image/jpeg")}


@app.route("/image/<int:item_id>/<kind>")
@login_required
def image(item_id, kind):
    if kind not in KINDS:
        abort(404)
    column, mime = KINDS[kind]
    row = db_one("SELECT " + column + " AS data FROM images WHERE id = ? AND user_id = ?", (item_id, g.user["id"]))
    if row is None:
        abort(404)
    return Response(row["data"], mimetype=mime)


@app.route("/download/<int:item_id>")
@login_required
def download(item_id):
    row = db_one("SELECT processed FROM images WHERE id = ? AND user_id = ?", (item_id, g.user["id"]))
    if row is None:
        abort(404)
    return Response(row["processed"], mimetype="image/png",
                    headers={"Content-Disposition": "attachment; filename=processed_%d.png" % item_id})


# =====================================================================
# BATCH PROCESSING + JOB HISTORY
# =====================================================================
@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    uid = g.user["id"]
    album_list = db_all("SELECT id, name FROM albums WHERE user_id = ? ORDER BY name", (uid,))
    f = {"title": "", "desc": "", "tags": "", "steps": ["grayscale"], "size": 300, "album_id": None}
    error = ""
    if request.method == "POST":
        f = read_form(uid)
        files = [x for x in request.files.getlist("images") if x and x.filename]
        if not files:
            error = "Please choose at least one image."
        elif len(files) > MAX_BATCH:
            error = "You can upload at most %d images per batch." % MAX_BATCH
        else:
            start = time.perf_counter()
            ok = 0
            failed = []
            for file in files:
                try:
                    img, orig_size = load_upload(file)
                    title = os.path.splitext(os.path.basename(file.filename))[0][:60] or "Untitled"
                    create_image(uid, title, "", f["tags"], f["album_id"], f["steps"], f["size"], img, orig_size)
                    ok += 1
                except Exception:
                    failed.append(file.filename[:30])
            ms = (time.perf_counter() - start) * 1000
            db_run("""INSERT INTO jobs (user_id, kind, total, ok, failed, pipeline, duration_ms, detail, created)
                      VALUES (?,?,?,?,?,?,?,?,?)""",
                   (uid, "batch", len(files), ok, len(failed), ",".join(f["steps"]), ms,
                    ("Failed: " + ", ".join(failed)) if failed else "All files processed", now_text()))
            log_activity("Batch job", "%d of %d images processed" % (ok, len(files)))
            return redirect(url_for("jobs", msg="Batch finished: %d of %d images processed." % (ok, len(files))))
    return page(BATCH, title="Batch upload", active="batch", error=error, f=f, album_list=album_list,
                sizes=SIZES, steps_map=STEPS, steps_val=f["steps"], max_batch=MAX_BATCH)


@app.route("/jobs")
@login_required
def jobs():
    rows = db_all("SELECT * FROM jobs WHERE user_id = ? ORDER BY id DESC LIMIT 50", (g.user["id"],))
    return page(JOBS, title="Jobs", active="jobs", rows=rows)


# =====================================================================
# ALBUMS (second entity with a one-to-many relation to images)
# =====================================================================
@app.route("/albums", methods=["GET", "POST"])
@login_required
def albums():
    uid = g.user["id"]
    error = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:30]
        if not name:
            error = "Please enter an album name."
        else:
            try:
                db_run("INSERT INTO albums (user_id, name, created) VALUES (?,?,?)", (uid, name, now_text()))
                log_activity("Created album", name)
                return redirect(url_for("albums", msg="Album created."))
            except sqlite3.IntegrityError:
                error = "You already have an album with that name."
    rows = db_all("SELECT a.id, a.name, a.created, COUNT(i.id) AS n FROM albums a LEFT JOIN images i "
                  "ON i.album_id = a.id WHERE a.user_id = ? GROUP BY a.id ORDER BY a.name", (uid,))
    return page(ALBUMS, title="Albums", active="albums", album_rows=rows, error=error)


@app.route("/albums/<int:album_id>/rename", methods=["POST"])
@login_required
def album_rename(album_id):
    name = request.form.get("name", "").strip()[:30]
    row = db_one("SELECT id FROM albums WHERE id = ? AND user_id = ?", (album_id, g.user["id"]))
    if row is None:
        abort(404)
    if not name:
        return redirect(url_for("albums", msg="Album name cannot be empty."))
    try:
        db_run("UPDATE albums SET name = ? WHERE id = ? AND user_id = ?", (name, album_id, g.user["id"]))
    except sqlite3.IntegrityError:
        return redirect(url_for("albums", msg="You already have an album with that name."))
    log_activity("Renamed album", name)
    return redirect(url_for("albums", msg="Album renamed."))


@app.route("/albums/<int:album_id>/delete", methods=["POST"])
@login_required
def album_delete(album_id):
    row = db_one("SELECT name FROM albums WHERE id = ? AND user_id = ?", (album_id, g.user["id"]))
    if row is None:
        abort(404)
    db_run("DELETE FROM albums WHERE id = ? AND user_id = ?", (album_id, g.user["id"]))
    log_activity("Deleted album", row["name"])
    return redirect(url_for("albums", msg="Album deleted. Its images were kept."))


# =====================================================================
# ALGORITHM LAB (seam carving + Huffman) for a single image
# =====================================================================
def png_b64(img):
    return base64.b64encode(to_bytes(img, "PNG")).decode("ascii")


@app.route("/lab/<int:item_id>")
@login_required
def lab(item_id):
    it = get_image(item_id)
    if it is None:
        abort(404)
    row = db_one("SELECT original, processed FROM images WHERE id = ?", (item_id,))
    try:
        pct = int(request.args.get("pct", 20))
    except ValueError:
        pct = 20
    if pct not in (10, 20, 30, 40):
        pct = 20

    seam = None
    if request.args.get("seam"):
        small = Image.open(io.BytesIO(row["original"])).convert("RGB")
        small.thumbnail((160, 160))
        w, h = small.size
        k = max(1, int(w * pct / 100))
        start = time.perf_counter()
        result, preview, done = seam_carve(small, k)
        ms = round((time.perf_counter() - start) * 1000, 1)
        squash = small.resize((w - done, h))
        seam = {"w": w, "h": h, "out_w": w - done, "k": done, "ms": ms, "cells": done * w * h,
                "preview": png_b64(preview), "result": png_b64(result), "squash": png_b64(squash)}

    huff = None
    huff_ms = 0
    if request.args.get("huff"):
        start = time.perf_counter()
        huff = huffman_report(Image.open(io.BytesIO(row["processed"])).convert("RGB"))
        huff_ms = round((time.perf_counter() - start) * 1000, 1)
    return page(LAB, title="Algorithm lab", active="gallery", it=it, pct=pct, seam=seam, huff=huff, huff_ms=huff_ms)


# =====================================================================
# SAMPLE DATA
# =====================================================================
WORDS_A = ["Sunset", "Mountain", "Ocean", "Forest", "City", "Desert", "River", "Winter", "Garden", "Night",
           "Sunrise", "Island", "Valley", "Harbor", "Meadow"]
WORDS_B = ["View", "Walk", "Lights", "Trip", "Memory", "Shot", "Scene", "Morning", "Road", "Sky"]
SAMPLE_TAGS = ["nature", "city", "travel", "abstract", "night", "water"]


@app.route("/seed", methods=["POST"])
@login_required
def seed():
    uid = g.user["id"]
    row = db_one("SELECT id FROM albums WHERE user_id = ? AND name = ?", (uid, "Samples"))
    if row:
        album_id = row["id"]
    else:
        album_id = db_run("INSERT INTO albums (user_id, name, created) VALUES (?,?,?)", (uid, "Samples", now_text()))
    gradient = Image.linear_gradient("L")
    keys = list(STEPS)
    for _ in range(30):
        c1 = tuple(random.randint(0, 120) for _ in range(3))
        c2 = tuple(random.randint(130, 255) for _ in range(3))
        base = ImageOps.colorize(gradient.resize((240, random.choice([140, 160, 180, 200]))), black=c1, white=c2)
        if random.random() < 0.5:
            base = base.rotate(90, expand=True)
        title = "%s %s %d" % (random.choice(WORDS_A), random.choice(WORDS_B), random.randint(1, 99))
        steps = [k for k in keys if k in random.sample(keys, random.randint(1, 3))]
        tags = ",".join(random.sample(SAMPLE_TAGS, 2))
        create_image(uid, title, "Auto-generated sample image.", tags, album_id, steps, random.choice(SIZES),
                     base.convert("RGB"), base.size)
    log_activity("Added sample data", "30 images")
    return redirect(url_for("index", msg="30 sample images added to the Samples album."))


# =====================================================================
# DAA BENCHMARK PAGE
# =====================================================================
def make_chart(sizes, results):
    W, H = 680, 320
    left, right, top, bottom = 62, 20, 16, 40
    pw, ph = W - left - right, H - top - bottom
    max_ms = max(max(v) for v in results.values()) or 1.0

    def X(i):
        return left + pw * i / (len(sizes) - 1)

    def Y(ms):
        return top + ph * (1 - ms / max_ms)

    parts = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="Running time chart" '
             'style="max-width:700px;background:#fff;border:1px solid #d6deeb;border-radius:10px">' % (W, H)]
    for frac in (0, 0.5, 1):
        y = Y(max_ms * frac)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e3e9f3"/>' % (left, y, W - right, y))
        parts.append('<text x="%d" y="%.1f" font-size="11" text-anchor="end" fill="#5b6b86">%.1f ms</text>'
                     % (left - 6, y + 4, max_ms * frac))
    for i, n in enumerate(sizes):
        parts.append('<text x="%.1f" y="%d" font-size="11" text-anchor="middle" fill="#5b6b86">n=%d</text>'
                     % (X(i), H - 14, n))
    for k, a in ALGOS.items():
        pts = " ".join("%.1f,%.1f" % (X(i), Y(v)) for i, v in enumerate(results[k]))
        parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.5"/>' % (pts, a["color"]))
        for i, v in enumerate(results[k]):
            parts.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (X(i), Y(v), a["color"]))
    parts.append("</svg>")
    return Markup("".join(parts))


@app.route("/analysis")
@login_required
def analysis():
    case = request.args.get("case", "random")
    if case not in ("random", "sorted", "reversed"):
        case = "random"
    sizes = [100, 250, 500, 1000, 1500]
    results = None
    chart = ""
    if request.args.get("run"):
        rng = random.Random(7)
        results = {k: [] for k in ALGOS}
        for n in sizes:
            data = [rng.randint(0, 100000) for _ in range(n)]
            if case == "sorted":
                data.sort()
            elif case == "reversed":
                data.sort(reverse=True)
            for k, a in ALGOS.items():
                start = time.perf_counter()
                a["fn"](data, lambda x: x)
                results[k].append((time.perf_counter() - start) * 1000)
        chart = make_chart(sizes, results)
    names = {"random": "random input", "sorted": "already sorted input", "reversed": "reverse sorted input"}
    return page(ANALYSIS, title="DAA analysis", active="analysis", algos=ALGOS, case=case,
                case_name=names[case], sizes=sizes, results=results, chart=chart,
                project_algos=PROJECT_ALGOS)


# =====================================================================
# ADMIN PANEL
# =====================================================================
@app.route("/admin")
@admin_required
def admin():
    users = db_all(
        "SELECT u.id, u.username, u.role, u.created, "
        "(SELECT COUNT(*) FROM images i WHERE i.user_id = u.id) AS n_images, "
        "(SELECT COALESCE(SUM(file_size), 0) FROM images i WHERE i.user_id = u.id) AS storage "
        "FROM users u ORDER BY u.id")
    totals = {
        "users": len(users),
        "images": sum(u["n_images"] for u in users),
        "storage_kb": round(sum(u["storage"] for u in users) / 1024, 1),
        "jobs": db_one("SELECT COUNT(*) AS n FROM jobs")["n"],
    }
    activity = db_all("SELECT username, action, detail, created FROM activity ORDER BY id DESC LIMIT 40")
    return page(ADMIN, title="Admin", active="admin", users=users, totals=totals, activity=activity)


@app.route("/admin/user/<int:uid>/toggle", methods=["POST"])
@admin_required
def admin_toggle(uid):
    if uid == g.user["id"]:
        abort(403)
    row = db_one("SELECT username, role FROM users WHERE id = ?", (uid,))
    if row is None:
        abort(404)
    new_role = "user" if row["role"] == "admin" else "admin"
    db_run("UPDATE users SET role = ? WHERE id = ?", (new_role, uid))
    log_activity("Changed role", "%s is now %s" % (row["username"], new_role))
    return redirect(url_for("admin", msg="%s is now %s." % (row["username"], new_role)))


@app.route("/admin/user/<int:uid>/delete", methods=["POST"])
@admin_required
def admin_delete(uid):
    if uid == g.user["id"]:
        abort(403)
    row = db_one("SELECT username FROM users WHERE id = ?", (uid,))
    if row is None:
        abort(404)
    db_run("DELETE FROM users WHERE id = ?", (uid,))
    log_activity("Deleted user", row["username"])
    return redirect(url_for("admin", msg="User %s and all their data were deleted." % row["username"]))


# =====================================================================
# REST API + HEALTH CHECK (cloud-style endpoints)
# =====================================================================
@app.route("/health")
def health():
    return jsonify(status="ok", time=now_text())


@app.route("/api/images")
def api_images():
    if g.user is None:
        return jsonify(error="login required"), 401
    items = user_images(g.user["id"])
    return jsonify(count=len(items), images=[
        {"id": i["id"], "title": i["title"], "album": i["album_name"], "tags": [t for t in i["tags"].split(",") if t],
         "pipeline": [s for s in i["pipeline"].split(",") if s], "size_kb": round(i["file_size"] / 1024, 1),
         "favorite": bool(i["favorite"]), "created": i["created"],
         "url": url_for("image", item_id=i["id"], kind="processed")} for i in items])


@app.route("/api/stats")
def api_stats():
    if g.user is None:
        return jsonify(error="login required"), 401
    s = build_stats(g.user["id"])
    return jsonify(images=s["n_images"], storage_kb=s["storage_kb"], albums=s["n_albums"],
                   favorites=s["n_fav"], jobs=s["n_jobs"],
                   operations={n: c for n, c, _ in s["ops"]})


if __name__ == "__main__":
    app.run(debug=True)