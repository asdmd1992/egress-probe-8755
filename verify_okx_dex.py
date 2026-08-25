#!/usr/bin/env python3
"""verify_okx_dex.py — 只读验证 OKX DEX API（Web3 Waas）凭据
用法: python3 verify_okx_dex.py <api_key> <api_secret> <passphrase>
只读 GET（报价查询为只读，绝不执行 swap/交易）。请求间随机 sleep 8-15s。
"""
import sys, time, hmac, hashlib, base64, json, random, urllib.request, urllib.error

BASE = "https://www.okx.com"
DEX_CHAIN = "1"  # Ethereum
ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"

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
        print("   error:", body[:200]); return
    try:
        j = json.loads(body)
        if isinstance(j, dict) and "code" in j:
            print(f"   code: {j.get('code')}  msg: {j.get('msg')}")
            d = j.get("data")
            if d:
                s = json.dumps(d, ensure_ascii=False)
                print(f"   data: {s[:400]}")
        else:
            print("   body:", body[:300])
    except Exception:
        print("   raw body:", body[:300])

def main():
    if len(sys.argv) < 4:
        print("usage: verify_okx_dex.py <api_key> <api_secret> <passphrase>")
        sys.exit(1)
    key, secret, passphrase = sys.argv[1], sys.argv[2], sys.argv[3]
    paths = [
        "/api/v5/dex/aggregator/supported/chain",
        "/api/v5/dex/aggregator/all-tokens?chainId=" + DEX_CHAIN,
        f"/api/v5/dex/aggregator/quote?chainId={DEX_CHAIN}&amount=1000000&fromTokenAddress={ETH}&toTokenAddress={USDT}",
    ]
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
