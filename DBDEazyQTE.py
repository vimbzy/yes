import cv2
import numpy as np
import math
import time
import os
import threading
import keyboard
from datetime import datetime
from mss import mss
from ultralytics import YOLO
import logging
import sys
from pathlib import Path

# ================= CONFIG =================
CROP_W, CROP_H = 200, 200
REGION = {
    "left":   (2560 - CROP_W) // 2,
    "top":    (1440 - CROP_H) // 2,
    "width":  CROP_W,
    "height": CROP_H,
}

HOME_DIR = Path.home()
LOG_DIR = HOME_DIR / "SCGALLERY"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, "detection_log.txt")
DEBUG_LOG_FILENAME = os.path.join(LOG_DIR, "detailed_debug_log.txt")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("--- System Started: Logging Active ---")

debug_logger = logging.getLogger("debug")
debug_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(DEBUG_LOG_FILENAME)
fh.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d | %(message)s', '%Y-%m-%d %H:%M:%S'))
debug_logger.addHandler(fh)

# ================= HELPERS =================
def grab_region(region):
    raw = sct.grab(region)
    frame = np.array(raw, dtype=np.uint8)
    frame = frame[:, :, :3]
    return np.ascontiguousarray(frame)

def angle_deg(x, y, cx, cy):
    return (math.degrees(math.atan2(y - cy, x - cx)) + 360.0) % 360.0

def unwrap_angle(curr, prev):
    """Keep red angle continuous across 0°/360° boundary."""
    if prev is None:
        return curr
    delta = curr - prev
    if delta < -180:
        delta += 360
    elif delta > 180:
        delta -= 360
    return prev + delta

# ================= SETTINGS =================
PRESS_KEY = "space"
DEBOUNCE_MS = 50
ANG_THRESH_DEG = 6.0
WHITE_ANG_MEMORY_S = 0.8

# ================= MODEL =================
MODEL_PATH = r"L:\yolov12\runs\detect\train2\weights\best.engine"
print(f"[system] loading model: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("[system] model loaded successfully.")
except Exception as e:
    print(f"[system] ERROR: could not load model. {e}")
    exit()

# ================= STATE =================
toggle = True
last_fire_ms = 0
_last_white_ang = None
_last_white_ang_t = 0.0
_last_red_unwrap = None
sct = mss()

# ================= CORE =================
def try_fire():
    """Press the key with minimal latency and log the event."""
    global last_fire_ms
    now_ms = time.time() * 1000
    if now_ms - last_fire_ms > DEBOUNCE_MS:
        logging.info(f"*** FIRE! (Δ={now_ms - last_fire_ms:.0f}ms) ***")
        keyboard.press(PRESS_KEY)
        time.sleep(0.015)
        keyboard.release(PRESS_KEY)
        last_fire_ms = now_ms

def keyboard_callback(event):
    global toggle
    key = event.name.lower()
    if key in ("f1", "caps lock"):
        toggle = not toggle
        logging.info(f"[toggle] running={toggle}")

def start_keyboard_listener():
    threading.Thread(target=lambda: keyboard.on_press(keyboard_callback), daemon=True).start()

# ================= MAIN LOOP =================
def run_system():
    print("[system] starting main loop…")
    start_keyboard_listener()
    global _last_white_ang, _last_white_ang_t, _last_red_unwrap

    # Window setup
    cv2.namedWindow("view", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("view", 800, 800)
    cv2.setWindowProperty("view", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    while True:
        loop_start = time.perf_counter()
        if not hasattr(run_system, "_white_history"):
            run_system._white_history = []

        # grab frame
        frame = grab_region(REGION)
        results = model(frame, verbose=False, imgsz=640, conf=0.15)
        boxes = results[0].boxes
        names = results[0].names

        red_ang, detected_white_ang = None, None
        cx, cy, R = None, None, None

        # find the circle (keyboard)
        for box in boxes:
            cls_name = names[int(box.cls[0])]
            if cls_name == "keyboard":
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                R = int(max(x2 - x1, y2 - y1) * 0.6)
                break

        # find red and white
        if cx is not None:
            for box in boxes:
                cls_name = names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0]
                obj_x, obj_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if cls_name == "red pin":
                    red_ang = angle_deg(obj_x, obj_y, cx, cy)
                elif cls_name == "great check":
                    detected_white_ang = angle_deg(obj_x, obj_y, cx, cy)

        # --- Maintain and smooth white angle ---
        t_now = time.perf_counter()

        if detected_white_ang is not None:
            _last_white_ang, _last_white_ang_t = detected_white_ang, t_now
            run_system._white_history.append(detected_white_ang)
            if len(run_system._white_history) > 5:  # keep last 5 frames
                run_system._white_history.pop(0)
        elif _last_white_ang is not None and (t_now - _last_white_ang_t < WHITE_ANG_MEMORY_S):
            run_system._white_history.append(_last_white_ang)
            if len(run_system._white_history) > 5:
                run_system._white_history.pop(0)
        else:
            detected_white_ang = None
            _last_white_ang = None
            run_system._white_history.clear()

        # smooth using recent history
        if run_system._white_history:
            detected_white_ang = float(np.mean(run_system._white_history))
        # -----------------------------------------

        # unwrap red angle
        if red_ang is not None:
            red_ang_unwrap = unwrap_angle(red_ang, _last_red_unwrap)
            _last_red_unwrap = red_ang_unwrap
        else:
            red_ang_unwrap = None

        debug_logger.debug(
            f"[detect] red_ang={red_ang} | white_ang={detected_white_ang} | cx={cx} cy={cy} R={R}"
        )

        # --- FIRE CONDITION (improved) ---
        if toggle and red_ang_unwrap is not None and detected_white_ang is not None:
            diff = abs((red_ang_unwrap - detected_white_ang) % 360)
            if diff > 180:
                diff = 360 - diff

            debug_logger.debug(
                f"[check] red={red_ang_unwrap:.2f} | white={detected_white_ang:.2f} | diff={diff:.2f}"
            )

            if diff <= ANG_THRESH_DEG:
                logging.info(
                    f"[FIRE TRIGGER] red={red_ang_unwrap:.2f} | white={detected_white_ang:.2f} | diff={diff:.2f}"
                )
                try_fire()
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1
                )
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            elif diff <= ANG_THRESH_DEG + 2:
                debug_logger.debug(
                    f"[NEAR-MISS] red={red_ang_unwrap:.2f} | white={detected_white_ang:.2f} | diff={diff:.2f}"
                )
        # ---------------------------------

        # HUD
        hud_color = (0, 255, 0) if toggle else (0, 255, 255)
        hud_text = (
            "[PAUSED]"
            if not toggle
            else f"Red:{red_ang_unwrap}  White:{detected_white_ang}"
        )

        # draw
        if cx is not None:
            cv2.circle(frame, (cx, cy), R, (255, 0, 0), 1)
            if red_ang is not None:
                rx = int(cx + R * math.cos(math.radians(red_ang)))
                ry = int(cy + R * math.sin(math.radians(red_ang)))
                cv2.line(frame, (cx, cy), (rx, ry), (0, 0, 255), 2)
            if detected_white_ang is not None:
                wx = int(cx + R * math.cos(math.radians(detected_white_ang)))
                wy = int(cy + R * math.sin(math.radians(detected_white_ang)))
                cv2.circle(frame, (wx, wy), 5, (255, 255, 255), -1)

        cv2.putText(
            frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2
        )
        cv2.imshow("view", frame)

        loop_time = (time.perf_counter() - loop_start) * 1000
        debug_logger.debug(f"[loop] {loop_time:.2f} ms ({1000/loop_time:.1f} FPS)")

        if cv2.waitKey(1) & 0xFF == 27:
            logging.info("--- System Shutdown ---")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()
