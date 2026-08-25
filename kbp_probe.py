#!/usr/bin/env python3
"""kbp_probe.py - diagnose reachability to KBitPay admin portal from runner egress.

Read-only: GET /api/webv1/Common/GetRsaPublicKey (and optional extra paths),
prints HTTP status + response excerpt. No email, no login, no writes.

Usage: python3 kbp_probe.py [path1] [path2] ...
Env:  KBP_BASE (default https://admin.kbitpay.com)
"""
import os, sys, time
import urllib.request, urllib.error, ssl

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
PATHS = sys.argv[1:] or ["/api/webv1/Common/GetRsaPublicKey"]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def probe(path):
    url = BASE + path
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", UA)
    req.add_header("Accept", "application/json, text/plain, */*")
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=30, context=ctx)
        data = r.read().decode("utf-8", "ignore")
        print(f"[RESP] {path} -> HTTP {r.status} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:400], flush=True)
        return r.status
    except urllib.error.HTTPError as e:
        data = e.read().decode("utf-8", "ignore")
        print(f"[RESP] {path} -> HTTP {e.code} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:400], flush=True)
        return e.code
    except Exception as e:
        print(f"[RESP] {path} -> ERR {e}", flush=True)
        return 0


def my_ip():
    for svc in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://ipv4.icanhazip.com"):
        try:
            r = urllib.request.urlopen(urllib.request.Request(svc, headers={"User-Agent": "curl/8"}), timeout=10)
            ip = r.read().decode().strip()
            if ip:
                return ip
        except Exception:
            continue
    return "unknown"


if __name__ == "__main__":
    print(f"[*] base={BASE}", flush=True)
    print(f"[IP] egress_ip={my_ip()}", flush=True)
    for p in PATHS:
        probe(p)
        time.sleep(2)
    print("[*] done", flush=True)
