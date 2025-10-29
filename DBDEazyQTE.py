import json
import time
import math
import os
import threading
import numpy as np
from mss import mss
from pynput import keyboard
import winsound
import cv2

class QTEBot:
    """
    An automation bot for Quick Time Events (QTEs) in games, primarily designed for Dead by Daylight.
    The bot captures a portion of the screen, processes it to detect visual cues for QTEs,
    and simulates keyboard inputs to successfully complete them.
    """
    def __init__(self, config_path='config.json'):
        """
        Initializes the QTEBot.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.load_config(config_path)
        self.keyboard_controller = keyboard.Controller()
        
        # --- NEW: LOAD TEMPLATES ---
        self.templates = {}
        self.template_sizes = {}
        
        template_files = {
            self.repair_speed: "repair_template.png",
            self.heal_speed: "heal_template.png",
            self.wiggle_speed: "wiggle_template.png"
        }

        for speed, file_name in template_files.items():
            template = cv2.imread(file_name, 0) # 0 = load as grayscale
            if template is None:
                print(f"!!! ERROR: Could not load {file_name}")
                self.templates[speed] = None
                self.template_sizes[speed] = (10, 10) # Dummy size
            else:
                self.templates[speed] = template
                self.template_sizes[speed] = template.shape[::-1] # (w, h)
        # ----------------------------
        
        self.last_im_a = None
        self.speed_now = self.repair_speed
        
        # --- Screenshot Toggle ---
        self.save_screenshots = False  # Toggle for saving screenshots
        
        # --- Thread-dependant variables, moved to run() ---
        self.sct = None
        self.monitor = None
        self.region = None
        self.circle_mask = None
        
        # --- UPDATED: Tuned Constants ---
        # Updated RED values based on your hsv(2, 94%, 84%)
        self.LOWER_RED = np.array([0, 100, 100])   # H, S, V
        self.UPPER_RED = np.array([10, 255, 255])  # H, S, V
        
        # --- REMOVED OLD CONSTANTS ---
        # self.LOWER_WHITE, self.UPPER_WHITE, and the BOX_..._RATIOs 
        # are no longer needed because we use template matching.


    def load_config(self, config_path):
        """
        Loads configuration from a JSON file.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

    def _calculate_region(self):
        """
        Calculates the screen region to capture based on the monitor resolution.
        """
        monitor_width = self.monitor['width']
        monitor_height = self.monitor['height']
        
        if monitor_height == 1600: self.crop_w, self.crop_h = 250, 250
        elif monitor_height == 1080: self.crop_w, self.crop_h = 150, 150
        elif monitor_height == 2160: self.crop_w, self.crop_h = 330, 330
        else: self.crop_w, self.crop_h = 200, 200

        return {
            "top": int((monitor_height - self.crop_h) / 2),
            "left": int((monitor_width - self.crop_w) / 2),
            "width": self.crop_w,
            "height": self.crop_h
        }

    def screenshot(self):
        """
        Captures a screenshot and converts it directly to OpenCV's BGR format.
        """
        sct_img = self.sct.grab(self.region)
        img_rgb = np.frombuffer(sct_img.rgb, dtype=np.uint8).reshape(sct_img.height, sct_img.width, 3)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def save_frame_image(self, im1):
        """
        Saves every screenshot taken in the main loop with a unique timestamped filename.
        """
        if not self.save_screenshots:
            return
            
        if not os.path.exists(self.imgdir):
            os.makedirs(self.imgdir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        filename = f"frame_{timestamp}_{millis:03d}.png"
        path = os.path.join(self.imgdir, filename)
        cv2.imwrite(path, im1)

    def save_qte_image(self, im1):
        """
        Saves the screenshot for every QTE attempt with a unique timestamped filename.
        """
        if not self.save_screenshots:
            return
            
        if not os.path.exists(self.imgdir):
            os.makedirs(self.imgdir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        filename = f"qte_{timestamp}_{millis:03d}.png"
        path = os.path.join(self.imgdir, filename)
        cv2.imwrite(path, im1)

    def find_red(self, hsv_image):
        """
        Finds the red pixels in the image using vectorized operations.
        Handles the "wrap-around" hue value for red in HSV.
        """
        # --- First red range (0-10) ---
        # This range is set to catch H(0-10), S(150-255), V(150-255)
        # Your value [1, 240, 214] fits perfectly in this window.
        lower_red1 = np.array([0, 100, 100])    # H, S, V
        upper_red1 = np.array([10, 255, 255])   # H, S, V
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        
        # --- Second red range (170-179) ---
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        
        # Combine the two red masks
        color_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply the circle mask
        mask = cv2.bitwise_and(color_mask, self.circle_mask)
        target_points = np.argwhere(mask > 0)
        
        if target_points.size == 0:
            return None
            
        r_i, r_j, max_d = self.find_thickest_point(hsv_image.shape, target_points)
        if max_d < 1:
            return None
        return (r_i, r_j, max_d)
    def find_thickest_point(self, shape, target_points):
        """
        Finds the thickest point in a cluster of target points.
        """
        target_map = np.zeros((shape[0], shape[1]), dtype=bool)
        target_points_list = target_points.tolist()
        for i, j in target_points_list:
            target_map[i, j] = True

        max_r = target_points_list[0]
        max_d = 0
        for i, j in target_points_list:
            for d in range(1, 20):
                if i + d >= shape[0] or j + d >= shape[1] or i - d < 0 or j - d < 0: break
                if target_map[i + d, j + d] and target_map[i - d, j - d] and target_map[i - d, j + d] and target_map[i + d, j - d]:
                    if d > max_d:
                        max_d = d
                        max_r = [i, j]
                else: break
        return (max_r[0], max_r[1], max_d)

    def find_square(self, bgr_image):
        """
        Finds the success zone using template matching and returns ONLY the
        center point. This is fast, simple, and all our new logic needs.
        """
        
        # --- 1. Get the correct template for the current mode ---
        current_template = self.templates.get(self.speed_now)
        if current_template is None:
            return None # No template for this mode
            
        template_w, template_h = self.template_sizes[self.speed_now]

        # --- 2. Prepare image ---
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # --- 3. Perform template matching ---
        res = cv2.matchTemplate(gray_image, current_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # --- 4. Process the result ---
        print(f"Template: {self.speed_now} | Match Score: {max_val:.2f}")

        # --- 5. Check if match is good ---
        # We use the 45% threshold from our last fix
        if max_val > 0.45: 
            
            center_y = max_loc[1] + template_h // 2
            center_x = max_loc[0] + template_w // 2
            
            # --- 6. MERCILESS STORM LOGIC IS GONE ---
            # The bot is now fast enough to handle the stream 
            # as individual skill checks.
            
            # --- 7. Normal Skill Check ---
            return (center_y, center_x) # Return (row, col)
        
        return None # No good match found

    def wiggle(self, t1, deg1, direction, im1):
        speed=self.wiggle_speed*direction
        target1, target2 = 270, 90
        delta_deg1 = (target1-deg1) % (direction*360)
        delta_deg2 = (target2-deg1) % (direction*360)
        predict_time = min(delta_deg1/speed ,delta_deg2/speed)
        click_time = t1 + predict_time - self.press_and_release_delay + self.delay_degree/abs(speed)
        delta_t = click_time-time.time() 
        
        if 0 > delta_t > -0.1:
            self.keyboard_controller.press(keyboard.Key.space)
            self.keyboard_controller.release(keyboard.Key.space)
            time.sleep(0.13)
            return 
        try:
            if delta_t > 0: time.sleep(delta_t)
            self.keyboard_controller.press(keyboard.Key.space)
            self.keyboard_controller.release(keyboard.Key.space)
            cv2.imwrite(os.path.join(self.imgdir,'log_wiggle.png'), im1)
            time.sleep(0.13)
        except (ValueError, IndexError): pass

    def timer(self, im1, t1):
        # Save every QTE attempt image
        self.save_qte_image(im1)

        if not self.toggle: return

        hsv1 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
        r1 = self.find_red(hsv1)
        if not r1: return # No needle

        deg1 = self.cal_degree(r1[0]-self.crop_h/2, r1[1]-self.crop_w/2)
        
        # We need a second screenshot to find the direction
        im2 = self.screenshot()
        hsv2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
        r2 = self.find_red(hsv2)
        if not r2: return 
        
        deg2 = self.cal_degree(r2[0]-self.crop_h/2, r2[1]-self.crop_w/2)
        if deg1 == deg2: return # Needle isn't moving
        
        direction = -1 if (deg2 - deg1) % 360 > 180 else 1
        
        # --- Handle Wiggle Separately ---
        if self.speed_now==self.wiggle_speed:
            return self.wiggle(t1,deg1,direction,im1)

        # --- Find Target ---
        # Note: We pass im1, NOT hsv1
        white_coords = self.find_square(im1)
        if not white_coords: return # No target

        # --- All Logic is Now Time-Based ---
        try:
            target_deg = self.cal_degree(white_coords[0]-self.crop_h/2, white_coords[1]-self.crop_w/2)
            
            if(self.hyperfocus):
                speed = direction * self.speed_now * (1 + 0.04 * self.focus_level)
            else:
                speed = direction * self.speed_now
            
            if abs(speed) < 1: return # Avoid division by zero

            delta_deg = (target_deg - deg1) % (direction * 360)
            
            predict_time = delta_deg / speed
            
            # This is the core calculation
            click_time = (t1 + predict_time 
                          - self.press_and_release_delay 
                          + self.delay_degree / abs(speed))

            # Calculate how long to sleep
            delta_t = click_time - time.time()
            

            # --- This is the new "hit" logic ---
            if delta_t > 0:
                time.sleep(delta_t)
            
            self.keyboard_controller.press(keyboard.Key.space)
            self.keyboard_controller.release(keyboard.Key.space)

            if(self.hyperfocus): 
                self.focus_level = min(6, (self.focus_level + 1))
            
            # Optional: Add a small cooldown to prevent double-presses
            time.sleep(0.1) 

        except (ValueError, IndexError, ZeroDivisionError) as e:
            print(f"Error in timer calculation: {e}")
            cv2.imwrite(os.path.join(self.imgdir,'log_error.png'), im1)

    def cal_degree(self, x, y):
        a = np.array([-1, 0])
        b = np.array([x, y])
        norm_b = np.linalg.norm(b)
        if norm_b == 0: return 0
        cos_theta = np.dot(a, b) / (np.linalg.norm(a) * norm_b)
        degree = np.rad2deg(np.arccos(cos_theta))
        return 360 - degree if b[1] < 0 else degree

    def on_press(self, key):
        if key == keyboard.Key.f1:
            self.keyboard_switch = not self.keyboard_switch
            self.toggle = self.keyboard_switch
            winsound.Beep(350 if self.keyboard_switch else 200, 500)
        
        if not self.keyboard_switch: return

        if key == keyboard.Key.caps_lock:
            self.toggle = not self.toggle
            winsound.Beep(350 if self.toggle else 200, 500)
        
        if not self.toggle: return
            
        try: k = key.char
        except AttributeError: k = key.name

        if k is None: return # Ignore keys we can't identify
            
        if k in 'wasd': self.focus_level = 0
        elif k == '3': self.speed_now = self.repair_speed; winsound.Beep(262,500)
        elif k == '4': self.speed_now = self.heal_speed; winsound.Beep(300,500)
        elif k == '5': self.speed_now = self.wiggle_speed; winsound.Beep(440,500)
        elif k == '6':
            self.hyperfocus = not self.hyperfocus
            winsound.Beep(350 if self.hyperfocus else 200, 500)
        elif k == '7':
            self.save_screenshots = not self.save_screenshots
            print(f"Screenshot saving {'ENABLED' if self.save_screenshots else 'DISABLED'}")
            winsound.Beep(600 if self.save_screenshots else 400, 300)
        elif k == '=': self.delay_degree += 2; winsound.Beep(460,500); print(f"Delay Degree: {self.delay_degree}")
        elif k == '-': self.delay_degree -= 2; winsound.Beep(500,500); print(f"Delay Degree: {self.delay_degree}")

    def run(self):
        if not os.path.exists(self.imgdir): os.mkdir(self.imgdir)
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        # --- MOVED INITIALIZATION HERE ---
        # Initialize MSS and region INSIDE the bot thread
        try:
            self.sct = mss()
            self.monitor = self.sct.monitors[1]
            self.region = self._calculate_region()
            
            # Pre-calculate the circle mask
            self.circle_mask = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)
            center = (int(self.crop_w / 2), int(self.crop_h / 2))
            radius = int(self.crop_h / 2)
            cv2.circle(self.circle_mask, center, radius, 255, -1)
        except Exception as e:
            print(f"Error during bot initialization: {e}")
            listener.stop()
            return
        # -----------------------------------

        print("-----------------------------------------")
        print("  QTE Bot Initialized - Ready")
        print("-----------------------------------------")
        print("  Hotkeys:")
        print("    F1        : Enable/Disable Bot")
        print("    CapsLock  : Pause/Resume Bot")
        print("    3         : Set speed to Repair")
        print("    4         : Set speed to Heal")
        print("    5         : Set speed to Wiggle")
        print("    6         : Toggle Hyperfocus Mode")
        print("    7         : Toggle Screenshot Saving")
        print("    = (+)     : Increase Hit Delay")
        print("    - (-)     : Decrease Hit Delay")
        print("-----------------------------------------")
        print("Press F1 to enable the bot.")
        try:
            while True:
                t = time.time()
                im_array = self.screenshot()
                self.save_frame_image(im_array)  # Save screenshot if toggled on
                self.timer(im_array, t)
        except KeyboardInterrupt:
            if self.last_im_a is not None:
                cv2.imwrite(os.path.join(self.imgdir, 'last_log.png'), self.last_im_a)
            listener.stop()
        except Exception as e:
            print(f"Bot crashed during main loop: {e}")
            if self.last_im_a is not None:
                cv2.imwrite(os.path.join(self.imgdir, 'last_log_crash.png'), self.last_im_a)
            listener.stop()


# --- GUI Section ---
import sys
import customtkinter as ctk

class BotThread(threading.Thread):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

    def run(self):
        try:
            self.bot.run()
        except Exception as e:
            print(f"Bot stopped: {e}")

    def stop(self):
        self.bot.toggle = False

class QTEBotGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DBD QTE Bot Controller")
        self.geometry("400x350")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.bot = None
        self.bot_thread = None

        self.status_var = ctk.StringVar(value="Bot not running")
        self.mode_var = ctk.StringVar(value="Repair")
        self.hyperfocus_var = ctk.BooleanVar(value=False)
        self.delay_pixel_var = ctk.IntVar(value=0)

        self._build_ui()
        self._load_config()
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _build_ui(self):
        frame = ctk.CTkFrame(self, corner_radius=15)
        frame.pack(padx=24, pady=24, fill="both", expand=True)

        ctk.CTkLabel(frame, text="DBD QTE Bot", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(10, 18))

        status_row = ctk.CTkFrame(frame, fg_color="transparent")
        status_row.pack(fill="x", pady=2)
        ctk.CTkLabel(status_row, text="Status:", width=80, anchor="w").pack(side="left")
        ctk.CTkLabel(status_row, textvariable=self.status_var, text_color="#1e90ff", anchor="w").pack(side="left")

        mode_row = ctk.CTkFrame(frame, fg_color="transparent")
        mode_row.pack(fill="x", pady=2)
        ctk.CTkLabel(mode_row, text="Mode:", width=80, anchor="w").pack(side="left")
        ctk.CTkLabel(mode_row, textvariable=self.mode_var, anchor="w").pack(side="left")

        ctk.CTkLabel(frame, text="Hyperfocus:").pack(anchor="w", pady=(12, 0))
        ctk.CTkSwitch(frame, variable=self.hyperfocus_var, text="", command=self.toggle_hyperfocus).pack(anchor="w")

        ctk.CTkLabel(frame, text="Delay Pixel:").pack(anchor="w", pady=(12, 0))
        ctk.CTkSlider(frame, from_=-20, to=20, number_of_steps=40, variable=self.delay_pixel_var, command=lambda v: self.update_delay_pixel()).pack(fill="x", padx=10)

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(pady=(24, 0), fill="x")
        self.start_btn = ctk.CTkButton(btn_row, text="Start Bot", command=self.start_bot, width=120)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = ctk.CTkButton(btn_row, text="Stop Bot", command=self.stop_bot, state="disabled", width=120)
        self.stop_btn.pack(side="left", padx=8)

        ctk.CTkButton(frame, text="Exit", command=self.on_exit, fg_color="#222", hover_color="#444").pack(pady=(18, 0))

        # Hotkey info
        ctk.CTkLabel(frame, text="Hotkeys (in-game):", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(18, 0))
        ctk.CTkLabel(frame, text="F1: Enable/Disable | CapsLock: Pause/Resume\n3: Repair | 4: Heal | 5: Wiggle | 6: Hyperfocus\n7: Save Screenshots | = / - : Adjust Hit Delay", font=ctk.CTkFont(size=11), text_color="#aaa").pack()

    def _load_config(self):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            self.hyperfocus_var.set(config.get("hyperfocus", False))
            self.delay_pixel_var.set(config.get("delay_pixel", 0))
        except Exception:
            pass

    def _save_config(self):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            config["hyperfocus"] = self.hyperfocus_var.get()
            config["delay_pixel"] = self.delay_pixel_var.get()
            with open("config.json", "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass

    def start_bot(self):
        if self.bot_thread and self.bot_thread.is_alive():
            return
        self._save_config()
        self.bot = QTEBot()
        self.bot.hyperfocus = self.hyperfocus_var.get()
        self.bot.delay_pixel = self.delay_pixel_var.get()
        self.bot.toggle = True
        self.bot_thread = BotThread(self.bot)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        self.status_var.set("Bot running")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.update_mode_label()

    def stop_bot(self):
        if self.bot:
            self.bot.toggle = False
        if self.bot_thread:
            self.bot_thread.stop()
        self.status_var.set("Bot stopped")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def update_mode_label(self):
        if not self.bot:
            self.mode_var.set("Repair")
            return
        speed = getattr(self.bot, "speed_now", self.bot.repair_speed)
        if speed == self.bot.repair_speed:
            self.mode_var.set("Repair")
        elif speed == self.bot.heal_speed:
            self.mode_var.set("Heal")
        elif speed == self.bot.wiggle_speed:
            self.mode_var.set("Wiggle")
        else:
            self.mode_var.set("Custom")

    def toggle_hyperfocus(self):
        if self.bot:
            self.bot.hyperfocus = self.hyperfocus_var.get()
        self._save_config()

    def update_delay_pixel(self):
        if self.bot:
            self.bot.delay_pixel = self.delay_pixel_var.get()
        self._save_config()

    def on_exit(self):
        self.stop_bot()
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = QTEBotGUI()
    app.mainloop()