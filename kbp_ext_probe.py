#!/usr/bin/env python3
"""kbp_ext_probe.py — KBitPay ExternalApi (kfiat/api/*) signed probe from GitHub runner egress.

Read-only verification that the merchant ExternalApi signature surface on api.kbitpay.com
is reachable from a fresh runner IP (sandbox egress was ALB/WAF 403 on all paths, fact 8884).

Signing (from kbitpayapi SignatureValidationFilter.ValidateExtAppApiRequest + SignHelper):
  headers: X-App-Key=<AppKey>, X-Timestamp=<unix ms>, X-Nonce=<unique>,
           X-Signature=base64(HMAC-SHA256(AppSecret, ts+nonce+body)).lower()
  body: JSON sorted by key (ordinal, recursive), JSON.NET Formatting.None; GET => ts+nonce only
Routes probed:
  GET  /kfiat/api/Crypto/getrsakey                (mode=keys)
  POST /kfiat/api/appv1/custody-wallet/balance    (mode=bal, read-only balance query on a
                                                   well-known public BTC address; no writes)

AppKey|AppSecret pairs come from env EXT_APPS (one pair per line, '|' separated),
or from positional args when a single pair is used.
Usage:
  EXT_APPS="key1|secret1
key2|secret2" python3 kbp_ext_probe.py keys [base_url]
  python3 kbp_ext_probe.py bal [base_url]
"""
import os, sys, time, json, hmac, hashlib, base64, uuid, random
import urllib.request, urllib.error, ssl

BASE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TARGET", "https://api.kbitpay.com")
BAL_ADDR = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # well-known public BTC address (read-only balance query)
BAL_BODY = {"ChainSymbol": "BTC", "Address": BAL_ADDR, "Protocol": "btc"}


def sort_json(o):
    if isinstance(o, dict):
        return {k: sort_json(o[k]) for k in sorted(o.keys())}
    if isinstance(o, list):
        return [sort_json(v) for v in o]
    return o


def build_sig(app_secret, ts_ms, nonce, body_str):
    msg = f"{ts_ms}{nonce}{body_str}".encode()
    return base64.b64encode(hmac.new(app_secret.encode(), msg, hashlib.sha256).digest()).decode().lower()


def call(method, path, app_key, app_secret, body_obj=None):
    ts = int(time.time() * 1000)
    nonce = uuid.uuid4().hex[:16]
    body_str = ""
    if body_obj is not None:
        body_str = json.dumps(sort_json(body_obj), separators=(",", ":"))
    sig = build_sig(app_secret, ts, nonce, body_str)
    url = BASE + path
    req = urllib.request.Request(url, method=method)
    req.add_header("X-App-Key", app_key)
    req.add_header("X-Timestamp", str(ts))
    req.add_header("X-Nonce", nonce)
    req.add_header("X-Signature", sig)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36")
    if body_obj is not None:
        req.add_header("Content-Type", "application/json")
        req.data = body_str.encode()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    t0 = time.time()
    print(f"[REQ] {method} {url} app={app_key[:6]}... ts={ts} nonce={nonce} body={body_str or '(none)'}", flush=True)
    try:
        r = urllib.request.urlopen(req, timeout=30, context=ctx)
        data = r.read().decode(errors="replace")
        print(f"[RESP] HTTP {r.status} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:1500], flush=True)
        return r.status, data
    except urllib.error.HTTPError as e:
        data = e.read().decode(errors="replace")
        print(f"[RESP] HTTP {e.code} ({time.time()-t0:.1f}s) len={len(data)}", flush=True)
        print(data[:1500], flush=True)
        return e.code, data
    except Exception as e:
        print(f"[RESP] ERR {e}", flush=True)
        return 0, str(e)


def get_apps():
    apps = []
    raw = os.environ.get("EXT_APPS", "")
    if raw:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            k, _, s = line.partition("|")
            apps.append((k.strip(), s.strip()))
    if not apps and len(sys.argv) > 4:
        apps.append((sys.argv[3], sys.argv[4]))
    return apps


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "keys"
    apps = get_apps()
    try:
        import urllib.request as _u
        ip = _u.urlopen("https://api.ipify.org", timeout=15).read().decode().strip()
    except Exception:
        ip = "unknown"
    print(f"runner_ip={ip} mode={mode} apps={len(apps)} base={BASE}", flush=True)
    if not apps:
        print("NO_APPS", flush=True)
        sys.exit(1)
    for i, (ak, sk) in enumerate(apps):
        if mode == "keys":
            call("GET", "/kfiat/api/Crypto/getrsakey", ak, sk)
        elif mode == "bal":
            call("POST", "/kfiat/api/appv1/custody-wallet/balance", ak, sk, BAL_BODY)
        else:
            print(f"unknown mode {mode}", flush=True)
            sys.exit(1)
        if i < len(apps) - 1:
            time.sleep(random.randint(8, 15))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
