#!/usr/bin/env python3
"""probe_okx_v6.py — 探测 OKX V6 API 端点结构（只读，无认证/带认证对照）
用法: python3 probe_okx_v6.py [api_key] [api_secret] [passphrase]
"""
import sys, time, hmac, hashlib, base64, json, random, urllib.request, urllib.error

BASE = "https://www.okx.com"

def sign(timestamp, method, path, body, secret):
    msg = timestamp + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

def call(api_key, secret, passphrase, path):
    req = urllib.request.Request(BASE + path)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    if api_key:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        sig = sign(ts, "GET", path, "", secret)
        req.add_header("OK-ACCESS-KEY", api_key)
        req.add_header("OK-ACCESS-SIGN", sig)
        req.add_header("OK-ACCESS-TIMESTAMP", ts)
        req.add_header("OK-ACCESS-PASSPHRASE", passphrase)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)

def main():
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    secret = sys.argv[2] if len(sys.argv) > 2 else ""
    passphrase = sys.argv[3] if len(sys.argv) > 3 else ""
    paths = [
        "/api/v6/public/time",
        "/api/v6/dex/aggregator/supported/chain",
        "/api/v6/dex/aggregator/all-tokens?chainId=1",
    ]
    for i, p in enumerate(paths):
        if i > 0:
            s = random.uniform(8, 15)
            print(f"  (sleep {s:.1f}s)")
            time.sleep(s)
        st, body = call(key, secret, passphrase, p)
        print(f"== {p} (auth={'yes' if key else 'no'}) -> HTTP {st}")
        print(f"   body: {body[:250]}")
    print("DONE")

if __name__ == "__main__":
    main()
