#!/usr/bin/env python3
"""verify_okx.py — 只读验证 OKX API v5 凭据（经 GitHub Actions runner 出口，新 IP）
用法: python3 verify_okx.py <api_key> <api_secret> <passphrase>
只读 GET，无任何资金/交易操作。请求间随机 sleep 8-15s（隐蔽要求）。
"""
import sys, time, hmac, hashlib, base64, json, random, urllib.request, urllib.error

BASE = "https://www.okx.com"

def sign(timestamp, method, path, body, secret):
    msg = timestamp + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

def call(api_key, secret, passphrase, path):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    sig = sign(ts, "GET", path, "", secret)
    req = urllib.request.Request(BASE + path)
    req.add_header("OK-ACCESS-KEY", api_key)
    req.add_header("OK-ACCESS-SIGN", sig)
    req.add_header("OK-ACCESS-TIMESTAMP", ts)
    req.add_header("OK-ACCESS-PASSPHRASE", passphrase)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return -1, str(e)

def report(path, st, body):
    print(f"== {path} -> HTTP {st}")
    if st == -1:
        print("   error:", body[:200])
        return
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "code" in j:
            print(f"   code: {j.get('code')}  msg: {j.get('msg')}")
            d = j.get("data")
            if d:
                print("   data[0] keys:", list(d[0].keys()) if isinstance(d, list) and isinstance(d[0], dict) else d)
                # 对 balance/asset 摘要关键字段（只读不操作）
                if isinstance(d, list) and isinstance(d[0], dict):
                    for k in ("totalEq", "details", "uid", "acctLv", "posMode", "isoMode"):
                        if k in d[0]:
                            v = d[0][k]
                            s = json.dumps(v, ensure_ascii=False)
                            print(f"   {k}: {s[:200]}")
        else:
            print("   body:", body[:300])
    except Exception as e:
        print("   raw body:", body[:300])

def main():
    if len(sys.argv) < 4:
        print("usage: verify_okx.py <api_key> <api_secret> <passphrase>")
        sys.exit(1)
    key, secret, passphrase = sys.argv[1], sys.argv[2], sys.argv[3]
    # 主判据 + 权限/身份扩展（全部只读 GET，共 3 请求）
    paths = ["/api/v5/account/balance", "/api/v5/account/config", "/api/v5/asset/balances"]
    for i, p in enumerate(paths):
        if i > 0:
            s = random.uniform(8, 15)
            print(f"  (sleep {s:.1f}s)")
            time.sleep(s)
        st, body = call(key, secret, passphrase, p)
        report(p, st, body)
    print("DONE")

if __name__ == "__main__":
    main()
