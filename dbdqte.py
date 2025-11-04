import cv2, numpy as np, math, time, os, threading, winsound, keyboard
from datetime import datetime
from mss import mss

# ================= CONFIG =================
CROP_W, CROP_H = 200, 200                          # capture window
REGION = {
    "left":  (2560 - CROP_W) // 2,                 # center of a 2560x1440 monitor
    "top":   (1440 - CROP_H) // 2,
    "width":  CROP_W,
    "height": CROP_H,
}
LOG_DIR = "SCGALLERY"
os.makedirs(LOG_DIR, exist_ok=True)

PRESS_KEY        = "space"                         # key to press on trigger
DEBOUNCE_MS      = 50                            # min ms between presses
EARLY_TRIGGER_S  = 0.050                          # fire this many seconds before the predicted overlap
ANG_THRESH_DEG   = 2.0                             # visual message only (not gating)
RING_TOL_PX      = 55                       # how close to ring radius for square
RED_BAND_PX      = 15                             # ring band thickness where we search for red

# Angular velocity sanity & smoothing
ANG_MIN, ANG_MAX = 50.0, 700.0                     # plausible |ω| range (deg/s)
EMA_ALPHA        = 0.35                            # smoothing for ω
WINDOW_MIN       = 1                                # minimum samples for robust ω

# Squirm targets (absolute angles)
TARGET_ANGLES_SQUIRM = [90.0, 270.0]

# ================ STATE ====================
mode            = "overlap"   # "overlap" or "squirm"
toggle          = True        # global run/pause
viper_focus     = False
focus_level     = 0           # WASD resets to 0
delay_pixel     = 0           # kept for API parity (unused in this solver)
last_fire_ms    = 0

# ω estimator state
_ema_omega      = None
_last_t         = None
_last_ang_unwrap= None
WINDOW          = []          # (t, ang_unwrap)

sct = mss()

# ================ HELPERS ==================
def beep(hz, ms):
    try: winsound.Beep(hz, ms)
    except: pass

def grab_region(region):
    raw = sct.grab(region)                 # BGRA
    frame = np.array(raw, dtype=np.uint8)  # copy + dtype
    frame = frame[:, :, :3]                # -> BGR
    return np.ascontiguousarray(frame)     # cv2-safe layout

def angle_deg(x, y, cx, cy):
    # 0° = +x axis; increase CCW; normalize to [0, 360)
    return (math.degrees(math.atan2(y - cy, x - cx)) + 360.0) % 360.0

def ang_diff(a, b):
    # signed shortest difference a-b in [-180, +180]
    return ((a - b + 180.0) % 360.0) - 180.0

def _unwrap_deg(curr, prev):
    # keep continuity across 0/360 using shortest signed diff
    d = ang_diff(curr, prev)
    return prev + d

def effective_early_trigger():
    # viper_focus tightens lead a bit (mimics “more focus = earlier compensation”)
    gain = 1.0 - min(0.04 * focus_level, 0.20) if viper_focus else 1.0
    return max(0.010, EARLY_TRIGGER_S * gain)

# ================ DETECTION =================
def detect_circle_once(gray):
    # find the ring (one-time)
    c = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=120, param2=40,
        minRadius=gray.shape[0]//4, maxRadius=gray.shape[0]//2
    )
    if c is None: return None
    x, y, r = np.uint16(np.around(c))[0, 0]
    return int(x), int(y), int(r)

def red_line_angle(frame_bgr, cx, cy, R, band=RED_BAND_PX):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170,120,120), (180,255,255))
    mask_red = cv2.bitwise_or(m1, m2)

    H, W = mask_red.shape
    yy, xx = np.indices((H, W))
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    ring = ((rr > R - band) & (rr < R + band)).astype(np.uint8) * 255

    mask = cv2.bitwise_and(mask_red, ring)
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None

    # farthest point on ring is usually the pointer tip
    idx = np.argmax((xs - cx)**2 + (ys - cy)**2)
    return angle_deg(xs[idx], ys[idx], cx, cy)

def white_square_angle(frame_bgr, cx, cy, R, tol_px=RING_TOL_PX):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_w = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_err = None, 1e9
    for c in cnts:
        if cv2.contourArea(c) < 50: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        rad_err = abs(math.hypot(x - cx, y - cy) - R)
        if rad_err < best_err and rad_err < tol_px:
            best_err, best = rad_err, (x, y)

    if best is None: return None
    return angle_deg(best[0], best[1], cx, cy)

# ============ ANGULAR VELOCITY (ω) ===========
def omega_update(red_angle_deg):
    """
    Returns (omega_ema_deg_per_s, omega_instant) or (None, None) until stable.
    Adapts automatically to any speed (no fixed profiles).
    """
    global _ema_omega, _last_t, _last_ang_unwrap, WINDOW

    t = time.perf_counter()
    if _last_t is None:
        _last_t = t
        _last_ang_unwrap = red_angle_deg
        WINDOW.clear()
        WINDOW.append((t, red_angle_deg))
        return None, None

    ang_unwrap = _unwrap_deg(red_angle_deg, _last_ang_unwrap)
    dt = t - _last_t
    if dt <= 0.0005:
        return _ema_omega, None

    inst = (ang_unwrap - _last_ang_unwrap) / dt  # deg/s, signed
    _last_t, _last_ang_unwrap = t, ang_unwrap

    if abs(inst) > 3 * ANG_MAX:  # guard spikes
        return _ema_omega, inst

    WINDOW.append((t, ang_unwrap))
    if len(WINDOW) >= WINDOW_MIN:
        t0 = WINDOW[0][0]
        xs = np.array([w[0] - t0 for w in WINDOW])
        ys = np.array([w[1] for w in WINDOW])
        denom = (xs**2).sum()
        slope = (xs * ys).sum() / denom if denom > 1e-6 else inst
    else:
        slope = inst

    _ema_omega = slope if _ema_omega is None else (EMA_ALPHA * slope + (1 - EMA_ALPHA) * _ema_omega)

    if abs(_ema_omega) < ANG_MIN * 0.4 or abs(_ema_omega) > ANG_MAX * 1.8:
        return None, inst

    return _ema_omega, inst

# ================ FIRING =====================
def try_fire():
    global last_fire_ms
    now_ms = time.time() * 1000
    if (now_ms - last_fire_ms) > DEBOUNCE_MS:
        keyboard.press_and_release(PRESS_KEY)
        beep(600, 60)
        last_fire_ms = now_ms

def run_squirm(red_angle, omega_ema):
    """Fire exactly at 90° and 270° using ω prediction, direction-agnostic."""
    if omega_ema is None or abs(omega_ema) < ANG_MIN:
        return
    early = effective_early_trigger()
    for target in TARGET_ANGLES_SQUIRM:
        delta = ang_diff(target, red_angle)
        t_hit = delta / omega_ema if omega_ema != 0 else 9e9
        # if we’re within early window, lead by (t_hit - early)
        if 0.0 <= t_hit <= (early + 0.010):
            lead = max(0.0, t_hit - early)
            if lead > 0: time.sleep(lead)
            try_fire()
            break

def run_overlap(red_angle, white_angle, omega_ema):
    """Fire when red overlaps white using ω prediction (works at any speed)."""
    if white_angle is None or omega_ema is None or abs(omega_ema) < ANG_MIN:
        return
    early = effective_early_trigger()
    delta = ang_diff(white_angle, red_angle)
    t_hit = delta / omega_ema if omega_ema != 0 else 9e9
    if 0.0 <= t_hit <= (early + 0.060):
        lead = max(0.0, t_hit - early)
        if lead > 0: time.sleep(lead)
        try_fire()

# ============ KEYBOARD CONTROLS =============
def keyboard_callback(event):
    global toggle, mode, viper_focus, focus_level, delay_pixel
    key = event.name.lower()

    # F1 / CAPS: global toggle (kept to match your old UX)
    if key == "f1" or key == "caps lock":
        toggle = not toggle
        beep(350 if toggle else 200, 120)
        print(f"[toggle] running={toggle}")
        return

    # WASD resets focus level (old behavior)
    if key in ("w","a","s","d"):
        focus_level = 0

    # 3 = Overlap (red → white). 5 = Squirm (90° & 270°)
    if key == "3":
        mode = "overlap"
        focus_level = 0
        beep(262, 150)
        print("[mode] OVERLAP (red→white)")

    elif key == "5":
        mode = "squirm"
        focus_level = 0
        beep(440, 150)
        print("[mode] SQUIRM (90° & 270°)")

    # 6 = Viper focus toggle (reduces early trigger a bit)
    elif key == "6":
        viper_focus = not viper_focus
        beep(350 if viper_focus else 200, 150)
        print(f"[focus] viper={viper_focus}")

    # +/- keep the same prints, though not used in solver
    elif key == "=":
        delay_pixel += 2
        beep(460, 100)
        print(f"[delay_pixel] +2 → {delay_pixel}")

    elif key == "-":
        delay_pixel = max(0, delay_pixel - 2)
        beep(500, 100)
        print(f"[delay_pixel] -2 → {delay_pixel}")

def start_keyboard_listener():
    threading.Thread(target=lambda: keyboard.on_press(keyboard_callback), daemon=True).start()
    print("[keyboard] listener active")

# ================= MAIN LOOP ================
def run_system():
    print("[system] starting…")
    start_keyboard_listener()

    circle_cached = None
    last_log_time = 0

    while True:
        frame = grab_region(REGION)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # One-time circle detection
        if circle_cached is None:
            #circle_cached = detect_circle_once(gray)
            #if circle_cached is None:
                #vis = frame.copy()
                #cv2.putText(vis, "No circle found", (10, 25),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                #cv2.imshow("view", vis)
                #if cv2.waitKey(1) & 0xFF == 27:
                    #print("[system] shutting down...")
                    #break
                time.sleep(0.01) # Sleep a little so it doesn't spam
                continue

        cx, cy, R = circle_cached
        red_ang = red_line_angle(frame, cx, cy, R)
        white_ang = white_square_angle(frame, cx, cy, R)

        # Update ω (angular velocity) using the red angle
        omega_ema, _omega_inst = (None, None)
        if red_ang is not None:
            omega_ema, _omega_inst = omega_update(red_ang)

        # Logic & HUD text
        if toggle and red_ang is not None:
            if mode == "overlap":
                run_overlap(red_ang, white_ang, omega_ema)
            elif mode == "squirm":
                run_squirm(red_ang, omega_ema)

            txt = f"Red:{red_ang:6.1f}"
            if white_ang is not None:
                txt += f"  White:{white_ang:6.1f}"
            if omega_ema is not None:
                txt += f"  ω:{omega_ema:6.1f}°/s"
            cv2.putText(frame, txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            msg = "[PAUSED]" if not toggle else "[NO DETECTION]"
            cv2.putText(frame, msg, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Ring outline
        cv2.circle(frame, (cx, cy), R, (255, 0, 0), 1)

        # Overlays only when angles exist (0.0 is valid!)
        if red_ang is not None:
            rx = int(cx + R * math.cos(math.radians(red_ang)))
            ry = int(cy + R * math.sin(math.radians(red_ang)))
            cv2.line(frame, (cx, cy), (rx, ry), (0, 0, 255), 2)

        if white_ang is not None:
            wx = int(cx + R * math.cos(math.radians(white_ang)))
            wy = int(cy + R * math.sin(math.radians(white_ang)))
            cv2.circle(frame, (wx, wy), 5, (255, 255, 255), -1)

        # Show frame
        # Resize for display only (makes the window 2x bigger)
        #display_frame = cv2.resize(frame, (CROP_W * 2, CROP_H * 2), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow("view", display_frame)
        #cv2.imshow("view", frame)
        #if cv2.waitKey(1) & 0xFF == 27:
            #print("[system] shutting down...")
            #break
            time.sleep(0.001)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
