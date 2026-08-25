#!/usr/bin/env python3
"""verify_covalent2.py — 验证多枚 Covalent key（只读）
用法: python3 verify_covalent2.py <key1> [key2 ...]
"""
import sys, os, time, json, random, urllib.request, urllib.error

BASE = "https://api.covalenthq.com/v1"

def call(key, path):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)

def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else [k for k in os.environ.get("COVALENT_KEYS","").splitlines() if k]
    if not keys:
        print("usage: verify_covalent2.py <key1> [key2 ...]")
        sys.exit(1)
    for idx, key in enumerate(keys):
        if idx > 0:
            time.sleep(random.uniform(8, 15))
        print(f"===== covalent key #{idx+1}: {key[:12]}... =====")
        st, body = call(key, "/chains/")
        print(f"== GET /v1/chains/ -> HTTP {st}")
        if st == 200:
            try:
                j = json.loads(body)
                items = j.get("data", {}).get("items", [])
                print(f"   VALID key: chains listed = {len(items)}")
            except Exception as e:
                print("   parse err:", e, body[:200])
        else:
            print(f"   NOT VALID: {body[:200]}")
    print("DONE")

if __name__ == "__main__":
    main()
