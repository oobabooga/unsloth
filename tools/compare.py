#!/usr/bin/env python3
"""Compare two llama_ab.py observation files and print a markdown verdict."""
import json
import sys


def load(p):
    with open(p) as fh:
        return json.load(fh)


def key(rec):
    return (rec.get("prompt"), rec.get("repeat"))


def main():
    title, base_p, head_p = sys.argv[1], sys.argv[2], sys.argv[3]
    base, head = load(base_p), load(head_p)

    print("### %s" % title)
    print()
    print("| | base | head |")
    print("|---|---|---|")
    print("| UMA env | `%s` | `%s` |" % (base["uma"], head["uma"]))
    print("| server ready | %s (%s) | %s (%s) |" % (
        base.get("server_ready"), base.get("server_note"),
        head.get("server_ready"), head.get("server_note")))
    print("| model | `%s` | `%s` |" % (base["gguf"].split("/")[-1], head["gguf"].split("/")[-1]))
    print("| bytes | %s | %s |" % (base.get("gguf_size"), head.get("gguf_size")))
    print()

    if not (base.get("server_ready") and head.get("server_ready")):
        print("**INCONCLUSIVE** - a state never became ready, so nothing was compared.")
        print()
        for name, st in (("base", base), ("head", head)):
            if not st.get("server_ready"):
                print("<details><summary>%s server log tail</summary>\n\n```\n%s\n```\n</details>\n"
                      % (name, st.get("log_tail", "")[-4000:]))
        return 1

    b = {key(r): r for r in base["results"]}
    h = {key(r): r for r in head["results"]}

    print("| prompt | rep | tokens equal | base tok/s | head tok/s | base text (first 70) | head text (first 70) |")
    print("|---|---|---|---|---|---|---|")
    same = True
    seen_any = False
    for k in sorted(b):
        if k not in h:
            continue
        seen_any = True
        rb, rh = b[k], h[k]
        tb, th = rb.get("tokens") or [], rh.get("tokens") or []
        eq = (tb == th) if (tb and th) else (rb.get("content") == rh.get("content"))
        same = same and eq
        def esc(s):
            return (s or "")[:70].replace("|", "\\|").replace("\n", " ")
        print("| %s | %s | %s | %s | %s | `%s` | `%s` |" % (
            k[0], k[1], "yes" if eq else "**NO**",
            round(rb.get("predicted_per_second") or 0, 1),
            round(rh.get("predicted_per_second") or 0, 1),
            esc(rb.get("content")), esc(rh.get("content"))))
    print()

    # self-consistency within each state
    for name, st in (("base", base), ("head", head)):
        by_prompt = {}
        for r in st["results"]:
            by_prompt.setdefault(r.get("prompt"), []).append(
                tuple(r.get("tokens") or []) or (r.get("content"),))
        for p, vals in sorted(by_prompt.items()):
            print("- %s / %s: %d repeat(s), %d distinct output(s)" % (name, p, len(vals), len(set(vals))))
    print()

    if not seen_any:
        print("**INCONCLUSIVE** - no comparable completions.")
        return 1
    if same:
        print("**IDENTICAL** - every prompt produced the same token ids in both states.")
    else:
        print("**DIVERGED** - at least one prompt produced different token ids. Full text:")
        print()
        for k in sorted(b):
            if k in h and (b[k].get("tokens") or []) != (h[k].get("tokens") or []):
                print("<details><summary>%s rep %s</summary>\n" % k)
                print("base:\n\n```\n%s\n```\n" % (b[k].get("content") or ""))
                print("head:\n\n```\n%s\n```\n" % (h[k].get("content") or ""))
                print("</details>\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
