#!/usr/bin/env python3
"""
TCaptcha solver for cloud.tencent.com password-recovery flow (appId 2042382584).
Modes:
  probe : open recover page, submit email, dump captcha frame HTML/screenshots/meta (no solving)
  solve : attempt solving (slidepuzzle: OpenCV gap; click: reference-template match), let page auto-submit
          sendRecoverEmail, capture ticket/randstr + response
  reset : open a reset link (input arg), set a new password via the page's own JS, report result
Usage:
  TC_MODE=probe python3 tcaptcha_solve.py
  TC_MODE=solve python3 tcaptcha_solve.py
  TC_MODE=reset python3 tcaptcha_solve.py '<reset_link>'
Env: TC_EMAIL (default admin@qqlink.com), TC_MAX (default 3), TC_HEADFUL=1 for headed mode (xvfb)
"""
import os, sys, time, json, re, random, io, base64
import urllib.request

import numpy as np
import cv2

EMAIL = os.environ.get("TC_EMAIL", "admin@qqlink.com")
MAX_CHALLENGES = int(os.environ.get("TC_MAX", "3"))
HEADFUL = os.environ.get("TC_HEADFUL", "") == "1"
MODE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TC_MODE", "probe")
RESET_LINK = sys.argv[2] if len(sys.argv) > 2 else ""
EVID = "evidence"
os.makedirs(EVID, exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def log(*a):
    print("[tc]", *a, flush=True)

def dl(url, timeout=20):
    """download bytes with browser UA"""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://cloud.tencent.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def save_json(name, obj):
    with open(os.path.join(EVID, name), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

# ---------------- browser ----------------
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

def make_ctx(pw):
    browser = pw.chromium.launch(
        headless=not HEADFUL,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox",
              "--disable-dev-shm-usage", "--lang=zh-CN", "--window-size=1280,900"],
    )
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 860},
        user_agent=UA, locale="zh-CN", timezone_id="Asia/Shanghai",
    )
    ctx.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = window.chrome || {runtime:{}};
        const _q = Object.getOwnPropertyDescriptor(Navigator.prototype, 'userAgent');
        Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
    """)
    return browser, ctx

def find_captcha_frame(page):
    for f in page.frames:
        u = f.url
        if "captcha.qq.com" in u or "tcaptcha" in u or "cap_union" in u:
            return f
    return None

def wait_captcha_frame(page, timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = find_captcha_frame(page)
        if f:
            return f
        time.sleep(0.5)
    return None

def shot(page, name):
    page.screenshot(path=os.path.join(EVID, name), full_page=False)

def human_delay(a=0.4, b=1.2):
    time.sleep(random.uniform(a, b))

# ---------------- challenge analysis ----------------
def analyze_frame(frame):
    """return dict with detected elements"""
    try:
        html = frame.content()
    except Exception as e:
        return {"error": str(e)}
    with open(os.path.join(EVID, "frame_latest.html"), "w") as f:
        f.write(html)
    info = {"len": len(html)}
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = re.sub(r"\s+", " ", txt)
    info["text"] = txt[:600]
    imgs = frame.eval_on_selector_all("img", "els => els.map(e => ({src:e.src, cls:e.className, w:e.naturalWidth, h:e.naturalHeight}))") if frame.query_selector("img") else []
    info["imgs"] = imgs
    btns = frame.eval_on_selector_all("button", "els => els.map(e => ({t:(e.innerText||'').trim(), cls:e.className}))") if frame.query_selector("button") else []
    info["buttons"] = btns
    # slide markers
    info["has_slide"] = bool(re.search(r"slide|drag|handle", html, re.I))
    info["has_click_inst"] = bool(re.search(r"请(点击|依次|选择|选出)|点击下图|找出", txt))
    return info

def refresh_challenge(frame):
    """click the refresh icon inside the captcha frame"""
    for sel in ["[class*='refresh']", "[class*='reset']", "[id*='refresh']", "a[class*='reload']"]:
        el = frame.query_selector(sel)
        if el:
            try:
                el.click()
                return True
            except Exception:
                pass
    return False

# ---------------- slide solver ----------------
def find_slide_elements(info, frame):
    """return (bg_img_el, piece_img_el, handle_el)"""
    bg = piece = handle = None
    for img in info.get("imgs", []):
        s = (img.get("cls") or "").lower() + (img.get("src") or "").lower()
        if "bg" in s or "back" in s:
            bg = img
        if "piece" in s or "jigsaw" in s or "puzzle" in s or "slice" in s:
            piece = img
    for sel in ["[class*='slider']", "[class*='handle']", "[id*='slide']", "[class*='drag-btn']", "[class*='tc-btn']"]:
        el = frame.query_selector(sel)
        if el and el.bounding_box():
            handle = el
            break
    return bg, piece, handle

def gap_dx(bg_bytes, piece_bytes=None):
    """return gap offset in bg-image pixels"""
    bg = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bg is None:
        return None
    if piece_bytes:
        pc = cv2.imdecode(np.frombuffer(piece_bytes, np.uint8), cv2.IMREAD_COLOR)
        if pc is not None and pc.shape[0] < bg.shape[0] and pc.shape[1] < bg.shape[1]:
            res = cv2.matchTemplate(bg, pc, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mxl = cv2.minMaxLoc(res)
            return int(mxl[0])
    # edge-based gap detection
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(g, 100, 200)
    col_density = edges.sum(axis=0) / max(1, edges.shape[0])
    # find widest band of high edge density in the middle 70% width
    h, w = edges.shape
    x0, x1 = int(w*0.15), int(w*0.85)
    best, best_w = 0, 0
    i = x0
    while i < x1:
        if col_density[i] > 1.2:
            j = i
            while j < x1 and col_density[j] > 1.2:
                j += 1
            if j - i > best_w:
                best_w = j - i
                best = i
            i = j
        else:
            i += 1
    if best_w > 0:
        return best + best_w // 2
    return None

def solve_slide(page, frame, info):
    bg, piece, handle = find_slide_elements(info, frame)
    if not handle:
        log("slide: no handle found")
        return False
    if not bg:
        log("slide: no bg image found")
        return False
    try:
        bg_bytes = dl(bg["src"])
        pc_bytes = dl(piece["src"]) if piece and piece.get("src") else None
    except Exception as e:
        log("slide: dl fail", e)
        return False
    dx = gap_dx(bg_bytes, pc_bytes)
    log("slide: gap dx(raw img px) =", dx)
    if dx is None:
        return False
    # scale to displayed px
    el = None
    for sel in ["img[class*='bg']", "[class*='bg'] img", "img[class*='action']", "img"]:
        el = frame.query_selector(sel)
        if el:
            break
    if not el:
        return False
    bb = el.bounding_box()
    if not bb:
        return False
    nw = bg.get("w") or 320
    scale = bb["width"] / nw if nw else 1.0
    dx_px = int(dx * scale)
    hb = handle.bounding_box()
    if not hb:
        return False
    sx = hb["x"] + hb["width"] / 2
    sy = hb["y"] + hb["height"] / 2
    tx = sx + dx_px
    page.mouse.move(sx, sy, steps=5)
    page.mouse.down()
    steps = random.randint(18, 30)
    for i in range(1, steps + 1):
        # ease-out + slight y jitter
        t = i / steps
        ease = 1 - (1 - t) ** 3
        page.mouse.move(sx + (tx - sx) * ease, sy + random.uniform(-1.2, 1.2), steps=2)
        time.sleep(random.uniform(0.012, 0.03))
    page.mouse.up()
    log("slide: dragged", dx_px, "px")
    return True

# ---------------- click solver ----------------
def match_multiscale(scene, ref, thresh=0.72):
    """return list of (cx, cy, score) in scene px"""
    hits = []
    for scale in [1.0, 0.9, 0.8, 1.15, 0.7, 1.3]:
        r = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if r.shape[0] >= scene.shape[0] or r.shape[1] >= scene.shape[1]:
            continue
        res = cv2.matchTemplate(scene, r, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mxl = cv2.minMaxLoc(res)
        if mx >= thresh:
            hits.append((mxl[0] + r.shape[1] / 2, mxl[1] + r.shape[0] / 2, mx))
    # NMS
    hits.sort(key=lambda h: -h[2])
    picked = []
    for h in hits:
        if all(np.hypot(h[0] - p[0], h[1] - p[1]) > 25 for p in picked):
            picked.append(h)
    return picked

def solve_click(page, frame, info):
    """click-type: find scene img + optional reference img; template-match & click"""
    imgs = info.get("imgs", [])
    if not imgs:
        log("click: no imgs")
        return False
    # scene = biggest img; ref = small img if any (sample/tip/demo class)
    scene_img = max(imgs, key=lambda i: (i.get("w") or 0) * (i.get("h") or 0))
    ref_img = None
    for i in imgs:
        s = (i.get("cls") or "").lower()
        if any(k in s for k in ("sample", "tip", "demo", "example", "ref", "small")):
            ref_img = i
            break
    log("click: scene", scene_img.get("src")[:120], "ref:", ref_img.get("src", "")[:80] if ref_img else None)
    if not ref_img:
        log("click: no reference image -> cannot template match")
        return False
    try:
        scene_b = dl(scene_img["src"])
        ref_b = dl(ref_img["src"])
    except Exception as e:
        log("click: dl fail", e)
        return False
    scene = cv2.imdecode(np.frombuffer(scene_b, np.uint8), cv2.IMREAD_COLOR)
    ref = cv2.imdecode(np.frombuffer(ref_b, np.uint8), cv2.IMREAD_COLOR)
    if scene is None or ref is None:
        log("click: decode fail")
        return False
    hits = match_multiscale(scene, ref)
    log("click: hits", [(int(x), int(y), round(s, 3)) for x, y, s in hits])
    if not hits:
        return False
    # map to displayed element coords
    el = None
    for sel in ["img", "canvas"]:
        el = frame.query_selector(sel)
        if el:
            break
    if not el:
        return False
    bb = el.bounding_box()
    if not bb:
        return False
    sw, sh = scene_img.get("w") or bb["width"], scene_img.get("h") or bb["height"]
    sx, sy = bb["width"] / sw, bb["height"] / sh
    for cx, cy, sc in hits[:4]:
        px = cx * sx
        py = cy * sy
        log("click at", round(px, 1), round(py, 1), "score", round(sc, 3))
        human_delay(0.25, 0.7)
        try:
            page.mouse.click(bb["x"] + px, bb["y"] + py)
        except Exception as e:
            log("click err", e)
    human_delay(0.5, 1.0)
    # confirm
    for sel in ["button", "[class*='confirm']", "[class*='submit']"]:
        els = frame.query_selector_all(sel)
        for e in els:
            t = (e.inner_text() or "").strip()
            if t in ("确认", "确定", "验证", "完成", "提交") or "confirm" in (e.get_attribute("class") or ""):
                try:
                    e.click()
                    log("click: confirm pressed")
                    return True
                except Exception:
                    pass
    return True

# ---------------- main flow ----------------
def do_probe(page):
    page.goto("https://cloud.tencent.com/account/password/recover", wait_until="networkidle", timeout=60000)
    page.wait_for_selector('input[name="email"]', timeout=30000)
    shot(page, "recover_page.png")
    page.fill('input[name="email"]', EMAIL)
    page.click('button[type="submit"]')
    time.sleep(3)
    frame = wait_captcha_frame(page, 25)
    if not frame:
        log("probe: no captcha frame appeared")
        shot(page, "no_captcha.png")
        return
    log("probe: captcha frame url:", frame.url)
    human_delay(2, 4)
    info = analyze_frame(frame)
    save_json("probe_info.json", info)
    shot(page, "captcha_view.png")
    try:
        fe = frame.frame_element()
        fe.screenshot(path=os.path.join(EVID, "captcha_frame.png"))
    except Exception as e:
        log("frame shot err", e)
    # prehandle meta
    log("probe done. summary:", json.dumps({k: info.get(k) for k in ("len", "text", "has_slide", "has_click_inst", "buttons")}, ensure_ascii=False)[:800])

def do_solve(page):
    page.goto("https://cloud.tencent.com/account/password/recover", wait_until="networkidle", timeout=60000)
    page.wait_for_selector('input[name="email"]', timeout=30000)
    captured = {"req": [], "resp": []}
    def on_req(r):
        if "sendRecoverEmail" in r.url:
            try:
                captured["req"].append({"url": r.url, "post": r.post_data})
            except Exception:
                pass
    def on_resp(r):
        if "sendRecoverEmail" in r.url:
            try:
                body = r.text()
            except Exception:
                body = ""
            captured["resp"].append({"url": r.url, "status": r.status, "body": body[:500]})
            log("SENDRECOVER RESP", r.status, body[:500])
    page.on("request", on_req)
    page.on("response", on_resp)
    page.fill('input[name="email"]', EMAIL)
    page.click('button[type="submit"]')
    time.sleep(3)
    frame = wait_captcha_frame(page, 25)
    if not frame:
        log("solve: no captcha frame")
        shot(page, "no_captcha.png")
        return
    log("solve: frame", frame.url)
    for attempt in range(1, MAX_CHALLENGES + 1):
        human_delay(1.5, 3)
        info = analyze_frame(frame)
        save_json(f"attempt_{attempt}_info.json", info)
        shot(page, f"attempt_{attempt}_view.png")
        try:
            frame.frame_element().screenshot(path=os.path.join(EVID, f"attempt_{attempt}_frame.png"))
        except Exception:
            pass
        txt = info.get("text", "")
        if info.get("has_slide"):
            log(f"attempt {attempt}: slide challenge")
            solve_slide(page, frame, info)
        elif info.get("has_click_inst") or "请" in txt:
            log(f"attempt {attempt}: click challenge")
            solve_click(page, frame, info)
        else:
            log(f"attempt {attempt}: unknown challenge type; text={txt[:200]}")
        # wait for outcome (ticket callback -> auto sendRecoverEmail) or failure toast
        t0 = time.time()
        solved = False
        while time.time() - t0 < 10:
            if captured["req"] or captured["resp"]:
                solved = True
                break
            # look for failure markers in frame
            time.sleep(1)
        if solved:
            log("solve: success (sendRecoverEmail fired)")
            break
        if attempt < MAX_CHALLENGES:
            log("attempt", attempt, "failed; refresh")
            if not refresh_challenge(frame):
                time.sleep(2)
                frame = wait_captcha_frame(page, 8) or frame
    save_json("captured.json", captured)
    log("solve: captured requests:", json.dumps(captured["req"], ensure_ascii=False))
    log("solve: captured responses:", json.dumps(captured["resp"], ensure_ascii=False))
    shot(page, "final_view.png")

def do_reset(page):
    if not RESET_LINK:
        log("reset: no link given")
        return
    page.goto(RESET_LINK, wait_until="networkidle", timeout=60000)
    time.sleep(3)
    shot(page, "reset_page.png")
    # dump inputs
    inputs = page.eval_on_selector_all("input", "els => els.map(e => ({name:e.name, type:e.type, id:e.id}))")
    log("reset: inputs", json.dumps(inputs, ensure_ascii=False))
    with open(os.path.join(EVID, "reset_page.html"), "w") as f:
        f.write(page.content())
    # try fill password fields if present
    pw = os.environ.get("TC_NEWPASS", "")
    if not pw:
        log("reset: TC_NEWPASS not set")
        return
    filled = False
    for sel in ["input[type='password']", "input[name*='password' i]", "input[name*='pwd' i]"]:
        els = page.query_selector_all(sel)
        for i, e in enumerate(els):
            try:
                e.fill(pw if i == 0 else pw)
                filled = True
            except Exception as ex:
                log("fill err", sel, i, ex)
    if filled:
        for sel in ["button[type='submit']", "button"]:
            for e in page.query_selector_all(sel):
                t = (e.inner_text() or "").strip()
                if t in ("确定", "确认", "提交", "完成", "重置", "设置"):
                    try:
                        e.click()
                        log("reset: submit pressed")
                        break
                    except Exception:
                        pass
        time.sleep(4)
        shot(page, "reset_after.png")
        log("reset: final url", page.url)
        log("reset: body head", re.sub(r"<[^>]+>", " ", page.content())[:400])
    else:
        log("reset: no password fields found")

def main():
    with sync_playwright() as pw:
        browser, ctx = make_ctx(pw)
        page = ctx.new_page()
        log("mode:", MODE, "email:", EMAIL, "headful:", HEADFUL)
        if MODE == "probe":
            do_probe(page)
        elif MODE == "solve":
            do_solve(page)
        elif MODE == "reset":
            do_reset(page)
        browser.close()

if __name__ == "__main__":
    main()
