# Provision desktop auth in a fresh Studio home and exchange the secret for tokens.
# Usage: mint.py <backend_dir> <studio_home> <port>
import json
import os
import sys
import urllib.request

backend, home, port = sys.argv[1:4]
os.environ["UNSLOTH_STUDIO_HOME"] = home
sys.path.insert(0, backend)
from auth import storage  # noqa: E402

secret = storage.create_desktop_secret()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/auth/desktop-login",
    data = json.dumps({"secret": secret}).encode(),
    headers = {"Content-Type": "application/json"},
)
body = json.load(urllib.request.urlopen(req))
print(json.dumps({k: body.get(k) for k in ("access_token", "refresh_token")}))
