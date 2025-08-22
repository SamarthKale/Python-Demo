from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import cv2
import mediapipe as mp
import threading
import time
import math
from collections import deque

# ---------- Window Setup ----------
from ursina import *

# ---------- Window Setup ----------
window.size = (864, 1536)
window.title = "Minecraft VR Cyberpunk"
window.borderless = False
window.fullscreen = False  # avoids fullscreen errors on laptops


# ---------- Ursina App ----------
app = Ursina()

# ---------- FPS Display ----------
fps_text = Text(text='FPS: 0', position=(-0.85,0.45), origin=(0,0), background=True, scale=1)

# ---------- Voxel Class ----------
class Voxel(Button):
    def __init__(self, position=(0,0,0), is_floor=False):
        super().__init__(
            parent=scene,
            position=position,
            model='cube',
            origin_y=.5,
            texture='white_cube',
            color=color.hsv(0, 0, random.uniform(.8, 1)) if not is_floor else color.brown,
            highlight_color=color.lime,
        )
        self.is_floor = is_floor

# ---------- Floor ----------
floor_blocks = []
for z in range(30):
    for x in range(30):
        voxel = Voxel(position=(x,0,z), is_floor=True)
        floor_blocks.append(voxel)

# ---------- Player ----------
player = FirstPersonController()
player.position = (15,1,15)
player.mouse_sensitivity = Vec2(0,0)  # Custom camera

# ---------- Cursor ----------
cursor = Entity(model='sphere', color=color.azure, scale=0.1, always_on_top=True)

# ---------- MediaPipe Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: Camera not available. Hand tracking disabled.")
    cap = None
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------- Thread Safety ----------
gesture_lock = threading.Lock()
right_hand_gesture = None
left_hand_gesture = None
aim_x, aim_y = 0, 0
running = True
action_cooldown = 0.1
last_action_time = 0

# ---------- Gesture Detection ----------
def get_hand_gesture(landmarks):
    fingers = []
    fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    tips = [8,12,16,20]
    pips = [6,10,14,18]
    for t,p in zip(tips,pips):
        fingers.append(1 if landmarks[t].y < landmarks[p].y else 0)
    total = sum(fingers)
    if total==0: return "fist"
    elif total==1 and fingers[1]==1: return "point"
    elif total==2 and fingers[1]==1 and fingers[2]==1: return "peace"
    elif total>=4: return "open"
    elif total==3 and fingers[1]==1 and fingers[2]==1 and fingers[3]==1: return "three"
    return "neutral"

# ---------- Hand Tracking Thread ----------
def hand_tracker():
    global right_hand_gesture, left_hand_gesture, aim_x, aim_y, running
    if not cap:
        return
    while running:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.01)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        current_right, current_left = None, None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                gesture = get_hand_gesture(hand_landmarks.landmark)
                if label=="Right":
                    current_right = gesture
                    tip = hand_landmarks.landmark[8]
                    with gesture_lock:
                        aim_x = (tip.x-0.5)*2
                        aim_y = (0.5-tip.y)*2
                else:
                    current_left = gesture
        with gesture_lock:
            right_hand_gesture = current_right
            left_hand_gesture = current_left
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# Start hand tracking
if cap:
    threading.Thread(target=hand_tracker, daemon=True).start()

# ---------- Movement ----------
movement_speed = 1.5
rotation_speed = 30
camera_rotation_x = 0
camera_rotation_y = 0

fps_counter = deque(maxlen=30)
def handle_controls():
    global left_hand_gesture, camera_rotation_x, camera_rotation_y
    # Update FPS
    fps_counter.append(time.time())
    if len(fps_counter)>1:
        fps_text.text = f"FPS: {len(fps_counter)/(fps_counter[-1]-fps_counter[0]):.1f}"
    with gesture_lock:
        gesture = left_hand_gesture
    if gesture=="fist": player.position += player.forward*movement_speed*time.dt
    elif gesture=="open": player.position -= player.forward*movement_speed*time.dt
    if gesture=="point": camera_rotation_y -= rotation_speed*time.dt
    elif gesture=="three": camera_rotation_x = max(camera_rotation_x-rotation_speed*time.dt, -90)
    elif gesture=="peace": camera_rotation_x = min(camera_rotation_x+rotation_speed*time.dt, 90)
    if held_keys['w']: player.position += player.forward*movement_speed*time.dt
    if held_keys['s']: player.position -= player.forward*movement_speed*time.dt
    if held_keys['a']: player.position -= player.right*movement_speed*time.dt
    if held_keys['d']: player.position += player.right*movement_speed*time.dt
    if held_keys['left arrow']: camera_rotation_y -= rotation_speed*time.dt
    if held_keys['right arrow']: camera_rotation_y += rotation_speed*time.dt
    if held_keys['up arrow']: camera_rotation_x = max(camera_rotation_x-rotation_speed*time.dt, -90)
    if held_keys['down arrow']: camera_rotation_x = min(camera_rotation_x+rotation_speed*time.dt, 90)
    player.rotation_y = camera_rotation_y
    camera.rotation_x = camera_rotation_x

# ---------- Update Loop ----------
def update():
    handle_controls()

def input(key):
    if key=='escape':
        global running
        running=False
        application.quit()

app.run()
