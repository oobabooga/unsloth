"""Test-only: open a CDP port on the main window. WebView2 ignores the env/policy overrides for this elevated session."""
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
w = d["app"]["windows"][0]
assert w["label"] == "main"
w["additionalBrowserArgs"] = "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --autoplay-policy=no-user-gesture-required --remote-debugging-port=9333 --lang=en-US --accept-lang=en-US"
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
print("patched", p)
