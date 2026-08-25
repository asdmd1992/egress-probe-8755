#!/usr/bin/env python3
"""probe_okx_ctl.py — OKX V6 DEX API 认证对照实验（只读）
用法: python3 probe_okx_ctl.py <api_key> <api_secret> <passphrase>
对照：真key / 假key / 无key 同一端点，判定 V6 DEX 端点是否认证保护、key 是否有效。
"""
import sys, time, hmac, hashlib, base64, json, random, urllib.request, urllib.error

BASE = "https://www.okx.com"
FAKE_KEY = "00000000-0000-0000-0000-000000000000"

def sign(timestamp, method, path, body, secret):
    msg = timestamp + method + path + body
    return base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

def call(key, secret, passphrase, path):
    req = urllib.request.Request(BASE + path)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    if key is not None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        sig = sign(ts, "GET", path, "", secret)
        req.add_header("OK-ACCESS-KEY", key)
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

def run_case(tag, key, secret, passphrase, path):
    st, body = call(key, secret, passphrase, path)
    print(f"== [{tag}] {path}")
    print(f"   HTTP {st} | {body[:220]}")
    time.sleep(random.uniform(8, 15))

def main():
    if len(sys.argv) < 4:
        print("usage: probe_okx_ctl.py <api_key> <api_secret> <passphrase>")
        sys.exit(1)
    key, secret, passphrase = sys.argv[1], sys.argv[2], sys.argv[3]
    ETH = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    all_tokens = "/api/v6/dex/aggregator/all-tokens?chainIndex=1"
    quote = (f"/api/v6/dex/aggregator/quote?chainIndex=1&amount=1000000"
             f"&fromTokenAddress={ETH}&toTokenAddress={USDT}")
    # 1. 真 key all-tokens
    run_case("real-key", key, secret, passphrase, all_tokens)
    # 2. 假 key all-tokens（认证保护对照）
    run_case("fake-key", FAKE_KEY, secret, passphrase, all_tokens)
    # 3. 无 key all-tokens（公开性对照）
    run_case("no-key", None, "", "", all_tokens)
    # 4. 真 key quote（只读报价，判定业务权限）
    run_case("real-key-quote", key, secret, passphrase, quote)
    # 5. 假 key quote
    run_case("fake-key-quote", FAKE_KEY, secret, passphrase, quote)
    print("DONE")

if __name__ == "__main__":
    main()
