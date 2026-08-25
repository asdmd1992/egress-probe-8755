#!/usr/bin/env python3
"""verify_covalent.py — 只读验证 Covalent API key（经 GitHub Actions runner 出口）
用法: python3 verify_covalent.py <api_key>
只读 GET。key 有效 -> /v1/chains 200；再抽验一个余额查询端点证明数据访问权。
"""
import sys, time, json, random, urllib.request, urllib.error

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
    if len(sys.argv) < 2:
        print("usage: verify_covalent.py <api_key>")
        sys.exit(1)
    key = sys.argv[1]
    st, body = call(key, "/chains/")
    print(f"== GET /v1/chains/ -> HTTP {st}")
    if st == 200:
        try:
            j = json.loads(body)
            items = j.get("data", {}).get("items", [])
            print(f"   VALID key: chains listed = {len(items)}")
            print("   chains sample:", [c.get("name") for c in items[:5]])
        except Exception as e:
            print("   parse err:", e, body[:200])
        # 数据访问权抽验（只读余额查询，USDT 合约地址）
        s = random.uniform(8, 15)
        print(f"  (sleep {s:.1f}s)")
        time.sleep(s)
        st2, body2 = call(key, "/1/address/0xdac17f958d2ee523a2206206994597c13d831ec7/balances_v2/")
        print(f"== GET /v1/1/address/<USDT>/balances_v2/ -> HTTP {st2}")
        if st2 == 200:
            try:
                j = json.loads(body2)
                it = j.get("data", {}).get("items", [])
                print(f"   data access OK: {len(it)} token balances, first: " +
                      json.dumps({k: it[0].get(k) for k in ("contract_name", "contract_ticker_symbol", "balance")}, ensure_ascii=False))
            except Exception as e:
                print("   parse err:", e, body2[:200])
        else:
            print("   body:", body2[:200])
    else:
        print(f"   INVALID or blocked: {body[:200]}")
    print("DONE")

if __name__ == "__main__":
    main()
