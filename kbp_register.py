#!/usr/bin/env python3
"""KBitPay mch/admin portal register+login driver (runs on GitHub Actions runner).

Register a NEW merchant account with an email verification code (type=1),
then login with the chosen password -> prints AccessToken.

Usage: python3 kbp_register.py <email> <pwd> <code>
Env:  KBP_BASE (default https://admin.kbitpay.com), KBP_APPID, KBP_ISADMIN
Low-rate: random 8-15s sleeps. Register sends no email (code already triggered
by the caller via SentVerificationCode type=1); Login is read-only.
"""
import base64, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbp_admin_login import Client

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
APPID = os.environ.get("KBP_APPID", "")
ISADMIN = os.environ.get("KBP_ISADMIN", "true").lower() != "false"


def pause():
    time.sleep(random.uniform(8, 15))


def main():
    if len(sys.argv) < 4:
        print("usage: kbp_register.py <email> <pwd> <code>", flush=True)
        sys.exit(1)
    em, pwd, code = sys.argv[1], sys.argv[2], sys.argv[3]
    c = Client()
    print("[*] base=%s appid=%r isAdmin=%s" % (BASE, APPID, ISADMIN), flush=True)
    c.fetch_key()
    print("[*] key ok KeyId=%s" % c.key_id, flush=True)

    # 1) CheckCaptchaCode (consume/validate the type=1 code)
    r0 = c.post_enc("/api/Register/CheckCaptchaCode",
                    {"email": em, "code": code, "type": "1"})
    print("REG CheckCaptchaCode -> %s" % json.dumps(r0, ensure_ascii=False)[:300], flush=True)
    pause()

    # 2) Register (SPA shape: pwd base64, verificationCode+code both set)
    body = {
        "email": em,
        "pwd": base64.b64encode(pwd.encode()).decode(),
        "password": base64.b64encode(pwd.encode()).decode(),
        "confirmPass": base64.b64encode(pwd.encode()).decode(),
        "verificationCode": code,
        "code": code,
        "type": 1,
        "isAdmin": ISADMIN,
        "rememberMe": False,
    }
    r1 = c.post_enc("/api/Register/Register", body)
    print("REG Register -> %s" % json.dumps(r1, ensure_ascii=False)[:300], flush=True)
    pause()

    # 3) Login with the fresh password
    r2 = c.post_enc("/api/Login/Login", {
        "appID": APPID,
        "pwd": base64.b64encode(pwd.encode()).decode(),
        "email": em, "isAdmin": ISADMIN, "language": "en"})
    rd = r2.get("ResData")
    tok = ""
    if isinstance(rd, dict):
        tok = rd.get("AccessToken") or rd.get("accessToken") or ""
    print("REG LOGIN -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
        r2.get("ErrCode"), r2.get("ErrMsg"), r2.get("Success"),
        (tok[:60] + "...") if tok else ""), flush=True)
    if tok:
        print("TOKEN %s" % tok, flush=True)
    print("[*] done", flush=True)


if __name__ == "__main__":
    main()
