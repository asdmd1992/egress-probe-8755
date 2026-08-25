#!/usr/bin/env python3
"""
tcprobe2.py — detailed in-page captcha probe for cloud.tencent.com recover flow.
Dumps: all frames, all img src/alt/class, visible text, network (captcha-domain) reqs+resps,
DOM snapshots, screenshots — every 2s for 60s after submitting the email.
Usage: TC_EMAIL=admin@qqlink.com python3 tcprobe2.py
"""
import os, sys, time, json, re, random
from playwright.sync_api import sync_playwright

EMAIL = os.environ.get("TC_EMAIL", "admin@qqlink.com")
EVID = "evidence"
os.makedirs(EVID, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def log(*a):
    print("[tc2]", *a, flush=True)

def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox",
                  "--disable-dev-shm-usage", "--lang=zh-CN"],
        )
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=UA, locale="zh-CN", timezone_id="Asia/Shanghai",
        )
        ctx.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        page = ctx.new_page()
        net = []

        def on_req(r):
            u = r.url
            if any(k in u for k in ("captcha", "cap_union", "tcaptcha", "tdc")):
                net.append({"req": u, "method": r.method})
                log("REQ", r.method, u[:220])

        def on_resp(r):
            u = r.url
            if any(k in u for k in ("cap_union", "captcha")):
                try:
                    body = r.text()
                except Exception:
                    body = ""
                net.append({"resp": u, "status": r.status, "len": len(body), "body": body[:4000]})
                log("RESP", r.status, u[:220], "len", len(body))
                if any(k in u for k in ("prehandle", "getcapbysig", "getcapby")):
                    with open(os.path.join(EVID, "captcha_api_" + re.sub(r"[^A-Za-z0-9]", "_", u)[-60:] + ".json"), "w") as f:
                        f.write(body)

        page.on("request", on_req)
        page.on("response", on_resp)

        log("goto recover")
        page.goto("https://cloud.tencent.com/account/password/recover",
                  wait_until="networkidle", timeout=60000)
        page.wait_for_selector('input[name="email"]', timeout=30000)
        page.screenshot(path=os.path.join(EVID, "step0_before.png"))
        page.fill('input[name="email"]', EMAIL)
        page.click('button[type="submit"]')

        snapshots = {}
        for i in range(30):  # 60s
            time.sleep(2)
            t = (i + 1) * 2
            frames = [f.url for f in page.frames]
            imgs = page.eval_on_selector_all(
                "img",
                "els => els.map(e=>({src:e.src, alt:e.alt, cls:e.className, w:e.naturalWidth, h:e.naturalHeight}))",
            )
            cimgs = [im for im in imgs if any(k in (im["src"] or "") for k in ("captcha", "cap_union", "tcaptcha", "ssl"))]
            try:
                txt = page.eval_on_selector("body", "e => e.innerText") or ""
            except Exception:
                txt = ""
            snapshots[t] = {"frames": frames, "cimgs": cimgs, "txt": txt[:1200]}
            if i % 5 == 0 or ("选择最符合" in txt) or cimgs:
                log(f"t={t}s frames={frames}")
                if cimgs:
                    log(f"t={t}s cimgs={json.dumps(cimgs, ensure_ascii=False)[:1500]}")
                if "选择最符合" in txt or "请" in txt[:200]:
                    log(f"t={t}s txt={txt[:500]}")
            if i in (3, 8, 15, 29):
                page.screenshot(path=os.path.join(EVID, f"step_{t}s.png"))
            if i == 15:
                with open(os.path.join(EVID, "dom_t30.html"), "w") as f:
                    f.write(page.content())

        with open(os.path.join(EVID, "net.json"), "w") as f:
            json.dump(net, f, ensure_ascii=False, indent=1)
        with open(os.path.join(EVID, "snapshots.json"), "w") as f:
            json.dump(snapshots, f, ensure_ascii=False, indent=1)
        # captcha-ish elements
        els = page.eval_on_selector_all(
            "[class*='captcha' i],[id*='captcha' i],[class*='tc-' i],[class*='verify' i]",
            "els => els.map(e=>({tag:e.tagName, id:e.id, cls:e.className, txt:(e.innerText||'').slice(0,300)}))",
        )
        with open(os.path.join(EVID, "captcha_els.json"), "w") as f:
            json.dump(els, f, ensure_ascii=False, indent=1)
        log("captcha els:", json.dumps(els, ensure_ascii=False)[:2500])
        page.screenshot(path=os.path.join(EVID, "final.png"))
        browser.close()

if __name__ == "__main__":
    main()
