#!/usr/bin/env python3
"""Parse claude --output-format stream-json and print human-readable progress."""
import sys
import json

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
        t = obj.get("type", "")
        if t == "assistant":
            for block in (obj.get("message") or {}).get("content", []):
                bt = (block or {}).get("type", "")
                if bt == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    hint = (
                        inp.get("query")
                        or inp.get("path")
                        or inp.get("file_path")
                        or str(inp)[:80]
                    )
                    print(f"  -> {name}: {str(hint)[:120]}", flush=True)
                elif bt == "text":
                    txt = (block.get("text") or "").strip()[:100].replace("\n", " ")
                    if txt:
                        print(f"  .. {txt}", flush=True)
        elif t == "result":
            turns = obj.get("num_turns", "?")
            cost = obj.get("total_cost_usd", "?")
            ok = not obj.get("is_error", False)
            status = "OK" if ok else "ERROR"
            print(f"  [{status}] turns={turns} cost=${cost}", flush=True)
    except Exception:
        pass
