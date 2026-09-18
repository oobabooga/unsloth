import json
import os
import sys
import urllib.parse
import urllib.request

_log = os.environ.get("LIVE_REQLOG")
_studio = os.environ.get("LIVE_STUDIO_DIR")
if _studio and _studio not in sys.path:
    sys.path.insert(0, _studio)

if _log:
    _orig = urllib.request.OpenerDirector.open

    def _open(self, fullurl, data = None, *a, **k):
        url = fullurl if isinstance(fullurl, str) else fullurl.full_url
        method = "GET" if isinstance(fullurl, str) else fullurl.get_method()
        try:
            with open(_log, "a", encoding = "utf-8") as f:
                f.write(json.dumps({"host": urllib.parse.urlsplit(url).netloc, "method": method,
                                    "path": urllib.parse.urlsplit(url).path[:160]}) + "\n")
        except Exception:
            pass
        return _orig(self, fullurl, data, *a, **k)

    urllib.request.OpenerDirector.open = _open
