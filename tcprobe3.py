#!/usr/bin/env python3
"""
tcprobe3.py — dump the captcha IFRAME rendered DOM (structure, imgs, text, bbox).
After submit, wait for #tcaptcha_iframe_dy, then dump everything inside it.
Usage: TC_EMAIL=admin@qqlink.com python3 tcprobe3.py
"""
import os, sys, time, json, re
from playwright.sync_api import sync_playwright

EMAIL = os.environ.get("TC_EMAIL", "admin@qqlink.com")
EVID = "evidence"
os.makedirs(EVID, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def log(*a):
    print("[tc3]", *a, flush=True)

def dump_frame(page, tag):
    frames = [f for f in page.frames if "gtimg.com" in f.url or "captcha" in f.url]
    out = {}
    for f in frames:
        try:
            html = f.content()
        except Exception as e:
            html = ""
        out[f.url] = html
        fn = os.path.join(EVID, f"frame_{tag}_{len(out)}.html")
        with open(fn, "w") as fh:
            fh.write(html)
        log(f"frame {tag}: {f.url} len={len(html)}")
        try:
            txt = f.eval_on_selector("body", "e => e.innerText") or ""
            log(f"  text: {txt[:600]}")
        except Exception:
            pass
        try:
            imgs = f.eval_on_selector_all(
                "img",
                "els => els.map(e=>({src:e.src, alt:e.alt, cls:e.className, w:e.naturalWidth, h:e.naturalHeight}))",
            )
            for im in imgs:
                log(f"  img: {json.dumps(im, ensure_ascii=False)[:300]}")
        except Exception:
            pass
        try:
            canv = f.eval_on_selector_all("canvas", "els => els.length")
            if canv:
                log("  canvases:", canv)
        except Exception:
            pass
        try:
            fe = f.frame_element()
            bb = fe.bounding_box()
            log("  frame bbox:", bb)
            fe.screenshot(path=os.path.join(EVID, f"frame_shot_{tag}_{len(out)}.png"))
        except Exception as e:
            log("  frame shot err:", e)
    return out

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
        page.goto("https://cloud.tencent.com/account/password/recover",
                  wait_until="networkidle", timeout=60000)
        page.wait_for_selector('input[name="email"]', timeout=30000)
        page.fill('input[name="email"]', EMAIL)
        page.click('button[type="submit"]')
        # wait for iframe
        ifr = None
        for i in range(30):
            time.sleep(2)
            try:
                ifr = page.wait_for_selector("#tcaptcha_iframe_dy", timeout=2000)
                if ifr:
                    log(f"iframe found at t={(i+1)*2}s")
                    break
            except Exception:
                pass
        time.sleep(4)
        dump_frame(page, "t1")
        page.screenshot(path=os.path.join(EVID, "page_view.png"))
        time.sleep(6)
        dump_frame(page, "t2")
        page.screenshot(path=os.path.join(EVID, "page_view2.png"))
        # main page captcha-ish elements w/ bbox
        els = page.eval_on_selector_all(
            "[class*='captcha' i],[id*='captcha' i],[class*='verify' i],[class*='tc-' i]",
            """els => els.map(e => {
                const r = e.getBoundingClientRect();
                return {tag:e.tagName, id:e.id, cls:e.className, x:r.x, y:r.y, w:r.width, h:r.height, txt:(e.innerText||'').slice(0,200)};
            })""",
        )
        with open(os.path.join(EVID, "main_captcha_els.json"), "w") as f:
            json.dump(els, f, ensure_ascii=False, indent=1)
        log("main els:", json.dumps(els, ensure_ascii=False)[:2000])
        browser.close()

if __name__ == "__main__":
    main()
