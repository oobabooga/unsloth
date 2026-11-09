"""Live A/B probe for PR 17206. Run with cwd = <tree>/studio/backend. Prints one JSON line per check."""
import importlib.util, json, os, subprocess, sys, time
from email.message import Message
from pathlib import Path

sys.path.insert(0, os.getcwd())
SAMPLE = Path(os.environ["PROBE_SAMPLE"]).read_bytes()

def emit(check, **kw):
    print("RESULT " + json.dumps({"check": check, **kw}), flush = True)

def present():
    return {n: importlib.util.find_spec(n) is not None for n in ("fitz", "pymupdf", "pymupdf4llm")}

def chat_export():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    import routes.chat_export as chat_export
    app = FastAPI(); app.include_router(chat_export.router, prefix = "/api/chat-export")
    app.dependency_overrides[get_current_subject] = lambda: "probe"
    t = time.monotonic()
    r = TestClient(app).post("/api/chat-export/pdf", json = {"title": "Probe", "html": "<p>Hello export</p>"})
    pdf = r.status_code == 200 and r.content.startswith(b"%PDF")
    detail = None if pdf else r.json().get("detail")
    emit("chat_export", status = r.status_code, pdf = pdf, detail = detail, seconds = round(time.monotonic() - t, 1))

def python_tool():
    from core.inference import tools
    t = time.monotonic()
    out = tools._python_exec("import fitz\nprint('FITZ_OK', fitz.version[0])", timeout = 60, session_id = "pdf-ab")
    emit("python_tool", ok = "FITZ_OK" in out, out = out.strip()[-300:], seconds = round(time.monotonic() - t, 1))

def web_pdf():
    from core.inference import tools
    class Resp:
        def __init__(self):
            self.body = SAMPLE; self.headers = Message(); self.headers["Content-Type"] = "application/pdf"
        def read(self, n = None):
            chunk, self.body = (self.body, b"") if n is None else (self.body[:n], self.body[n:])
            return chunk
    tools._validate_and_resolve_host = lambda host, port: (True, "", ["93.184.216.34"])
    opener = type("O", (), {"open": lambda self, req, timeout = None: Resp()})()
    tools.urllib.request.build_opener = lambda *a, **k: opener
    t = time.monotonic()
    out = tools._fetch_page_text("https://example.com/paper.pdf", timeout = float(os.environ.get("PROBE_WEB_TIMEOUT", "60")))
    emit("web_pdf", ok = "Live AB probe" in out, out = out.strip()[:300], seconds = round(time.monotonic() - t, 1))

def rag_parse():
    from core.rag.parsers import parse_pdf_bytes
    t = time.monotonic()
    try:
        pages, total = parse_pdf_bytes(SAMPLE)
        emit("rag_parse", ok = any("Live AB probe" in p.text for p in pages), seconds = round(time.monotonic() - t, 1))
    except Exception as exc:
        emit("rag_parse", ok = False, error = f"{type(exc).__name__}: {exc}"[:300])

def plain_child():
    r = subprocess.run([sys.executable, "-c", "import fitz, pymupdf4llm; print('CHILD_OK')"], capture_output = True, text = True)
    emit("plain_child_import", ok = "CHILD_OK" in r.stdout, err = r.stderr.strip()[-160:])

def parallel_exports():
    from concurrent.futures import ThreadPoolExecutor
    import routes.chat_export as chat_export
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda i: chat_export.render_chat_pdf_in_child(f"t{i}", f"<p>{i}</p>")[:4], range(2)))
    emit("parallel_exports", ok = results == [b"%PDF"] * 2)

def pip_fallback():
    from utils import optional_pdf
    optional_pdf._find_uv = lambda: None
    try:
        optional_pdf.load_pdf_module("fitz")
        emit("pip_fallback", ok = True)
    except Exception as exc:
        emit("pip_fallback", ok = False, error = str(exc)[:200], cause = repr(exc.__cause__)[:300])

def sandbox_engaged():
    from core.inference import tools
    workspace = os.environ.get("GITHUB_WORKSPACE", "/nonexistent")
    out = tools._python_exec(f"import os; print('HOST_REPO_VISIBLE', os.path.exists({workspace!r}))", timeout = 60, session_id = "pdf-ab")
    emit("sandbox_engaged", out = out.strip()[-120:])

emit("before", present = present())
for step in sys.argv[1:]:
    {"export": chat_export, "tool": python_tool, "web": web_pdf, "rag": rag_parse, "child": plain_child, "parallel": parallel_exports, "pip": pip_fallback, "sandbox": sandbox_engaged}[step]()
emit("after", present = present())
