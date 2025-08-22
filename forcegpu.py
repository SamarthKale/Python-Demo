# Add this BEFORE importing ursina to fix graphics issues
import os
os.environ['PANDA_DISPLAY_ERROR'] = '0'  # Hide graphics errors
os.environ['PANDA_GL_FORCE_SOFTWARE'] = '1'  # Force software rendering if needed

from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import cv2
import mediapipe as mp
import math
import threading
import time
from collections import deque

print("Starting Minecraft VR with graphics fixes...")
print("Forcing software rendering to fix grey screen issue...")

# Create app with specific graphics settings
app = Ursina()

# Force basic graphics settings
window.title = "Minecraft VR - 3D Game"
window.borderless = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Try to fix grey screen with explicit camera settings
camera.position = (15, 2, 15)  # Make sure camera is in a good position
camera.rotation = (0, 0, 0)    # Reset rotation
camera.fov = 90

# Force a simple background color (not grey)
camera.clear_color = color.cyan  # Sky blue background

print("✓ Graphics settings applied")
print("✓ Background should now be cyan (blue), not grey")

# ---------- Enhanced Lighting (might fix grey screen) ----------
# Add directional light
sun = DirectionalLight()
sun.look_at(Vec3(1, -1, -1))

# Add ambient light
AmbientLight(color=color.rgba(100, 100, 100, 0.1))

print("✓ Lighting added to scene")

# ---------- Test Objects First ----------
# Add some bright, obvious test objects to see if rendering works
test_cube1 = Entity(
    model='cube', 
    color=color.red, 
    position=(15, 2, 18),  # In front of player
    scale=2
)

test_cube2 = Entity(
    model='cube', 
    color=color.green, 
    position=(17, 2, 18),
    scale=2
)

test_cube3 = Entity(
    model='cube', 
    color=color.yellow, 
    position=(13, 2, 18),
    scale=2
)

# Big text in 3D space
test_text_3d = Text3d(
    "MINECRAFT VR WORKING!",
    position=(15, 4, 20),
    scale=3,
    color=color.white,
    billboard=True
)

print("✓ Test objects created - you should see 3 colored cubes and text")

# ---------- Game Blocks ----------
class Voxel(Button):
    def __init__(self, position=(0,0,0), is_floor=False):
        super().__init__(
            parent=scene,
            position=position,
            model='cube',
            origin_y=.5,
            color=color.brown if is_floor else color.white,  # Simplified colors
            highlight_color=color.lime,
        )
        self.is_floor = is_floor

print("Creating simplified floor...")
# Create smaller floor for testing (10x10 instead of 30x30)
floor_blocks = []
for z in range(10, 20):  # Smaller area around player
    for x in range(10, 20):
        voxel = Voxel(position=(x, 0, z), is_floor=True)
        floor_blocks.append(voxel)

print("✓ Floor created")

# Place player
player = FirstPersonController()
player.position = (15, 1, 15)
player.mouse_sensitivity = Vec2(0, 0)

# Make sure player is visible/working
player_indicator = Entity(
    model='cube',
    color=color.blue,
    position=(15, 0.5, 15),
    scale=0.5
)

print("✓ Player created at position:", player.position)

# Custom camera rotation variables
camera_rotation_x = 0
camera_rotation_y = 0
vertical_look_speed = 15

# Bright, obvious cursor
cursor = Entity(
    model='sphere', 
    color=color.magenta, 
    scale=0.3,  # Bigger
    always_on_top=True
)

print("✓ All game objects created")

# ---------- Simplified Hand Tracking (keep original) ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_CONTRAST, 128)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
        cap.set(cv2.CAP_PROP_HUE, 0)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        print("✓ Camera initialized")
except Exception as e:
    print(f"Camera error: {e}")
    cap = None

# Shared variables
gesture_lock = threading.Lock()
right_hand_gesture = None
left_hand_gesture = None
aim_x, aim_y = 0, 0
running = True
action_cooldown = 0.1
last_action_time = 0

def get_hand_gesture(landmarks):
    fingers = []
    
    if landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    total_fingers = sum(fingers)
    
    if total_fingers == 0:
        return "fist"
    elif total_fingers == 1 and fingers[1] == 1:
        return "point"
    elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
        return "peace"
    elif total_fingers >= 4:
        return "open"
    elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
        return "three"
    else:
        return "neutral"

def hand_tracker():
    global right_hand_gesture, left_hand_gesture, aim_x, aim_y, running
    
    if not cap:
        print("No camera - using keyboard only")
        return
    
    cv2.namedWindow("Hand Control - Natural View", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Hand Control - Natural View", 100, 100)
    
    frame_count = 0
    
    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            if frame_count % 2 == 0:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            h, w, _ = frame.shape
            current_right_gesture = None
            current_left_gesture = None
            
            # Visual zones
            cv2.rectangle(frame, (0, 0), (w//2, h), (100, 100, 255), 3)
            cv2.rectangle(frame, (w//2, 0), (w, h), (255, 100, 100), 3)
            cv2.putText(frame, "LEFT HAND ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            cv2.putText(frame, "RIGHT HAND ZONE", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            
            # Status
            cv2.putText(frame, "Should see colored cubes in 3D window!", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmarks = hand_landmarks.landmark
                    hand_label = handedness.classification[0].label
                    is_right_hand = (hand_label == "Right")
                    gesture = get_hand_gesture(landmarks)
                    
                    if is_right_hand:
                        current_right_gesture = gesture
                        index_tip = landmarks[8]
                        with gesture_lock:
                            aim_x = (index_tip.x - 0.5) * 2
                            aim_y = (0.5 - index_tip.y) * 2
                        
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        if gesture == "fist":
                            cv2.circle(frame, (ix, iy), 25, (0, 255, 0), -1)
                            cv2.putText(frame, "PLACE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        elif gesture == "peace":
                            cv2.circle(frame, (ix, iy), 25, (0, 0, 255), -1)
                            cv2.putText(frame, "DESTROY", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        elif gesture == "point":
                            cv2.circle(frame, (ix, iy), 15, (255, 255, 0), -1)
                            cv2.putText(frame, "AIMING", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                        else:
                            cv2.circle(frame, (ix, iy), 10, (255, 255, 255), 2)
                    else:
                        current_left_gesture = gesture
                        hand_center = landmarks[9]
                        hx, hy = int(hand_center.x * w), int(hand_center.y * h)
                        
                        if gesture == "fist":
                            cv2.circle(frame, (hx, hy), 25, (0, 255, 255), -1)
                            cv2.putText(frame, "FORWARD", (w//2 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        elif gesture == "open":
                            cv2.circle(frame, (hx, hy), 25, (255, 0, 255), -1)
                            cv2.putText(frame, "BACKWARD", (w//2 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                        elif gesture == "point":
                            cv2.circle(frame, (hx, hy), 25, (255, 165, 0), -1)
                            cv2.putText(frame, "TURN", (w//2 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
                        elif gesture == "three":
                            cv2.circle(frame, (hx, hy), 25, (0, 255, 0), -1)
                            cv2.putText(frame, "LOOK UP", (w//2 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        elif gesture == "peace":
                            cv2.circle(frame, (hx, hy), 25, (255, 0, 0), -1)
                            cv2.putText(frame, "LOOK DOWN", (w//2 + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        else:
                            cv2.circle(frame, (hx, hy), 15, (255, 255, 255), 2)
                    
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            with gesture_lock:
                right_hand_gesture = current_right_gesture
                left_hand_gesture = current_left_gesture
            
            cv2.imshow("Hand Control - Natural View", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
                break
                
        except Exception as e:
            print(f"Hand tracking error: {e}")
            time.sleep(0.1)
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# Start hand tracking
if cap:
    threading.Thread(target=hand_tracker, daemon=True).start()

# ---------- Controls ----------
movement_speed = 1.5
rotation_speed = 30

def handle_controls():
    global left_hand_gesture, camera_rotation_x, camera_rotation_y
    
    with gesture_lock:
        current_left_gesture = left_hand_gesture
    
    # Hand gestures
    if current_left_gesture == "fist":
        player.position += player.forward * movement_speed * time.dt
    elif current_left_gesture == "open":
        player.position -= player.forward * movement_speed * time.dt
    elif current_left_gesture == "point":
        camera_rotation_y -= rotation_speed * time.dt
    elif current_left_gesture == "three":
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
    elif current_left_gesture == "peace":
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Keyboard
    if held_keys['w']:
        player.position += player.forward * movement_speed * time.dt
    if held_keys['s']:
        player.position -= player.forward * movement_speed * time.dt
    if held_keys['a']:
        player.position -= player.right * movement_speed * time.dt
    if held_keys['d']:
        player.position += player.right * movement_speed * time.dt
    if held_keys['left arrow']:
        camera_rotation_y -= rotation_speed * time.dt
    if held_keys['right arrow']:
        camera_rotation_y += rotation_speed * time.dt
    if held_keys['up arrow']:
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
    if held_keys['down arrow']:
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Apply rotations
    player.rotation_y = camera_rotation_y
    camera.rotation_x = camera_rotation_x

# ---------- Block System ----------
keyboard_place_action = False
keyboard_destroy_action = False

def update():
    global right_hand_gesture, last_action_time, keyboard_place_action, keyboard_destroy_action
    
    handle_controls()
    
    current_time = time.time()
    
    with gesture_lock:
        current_gesture = right_hand_gesture
    
    if held_keys['space']:
        keyboard_place_action = True
    if held_keys['x']:
        keyboard_destroy_action = True
    
    # Simplified raycasting
    ray_origin = camera.world_position
    ray_direction = camera.forward
    
    hit_info = raycast(ray_origin, ray_direction, distance=10, ignore=[player, cursor])
    
    if hit_info.hit:
        cursor.position = hit_info.world_point
        cursor.visible = True
        
        # Block placement
        if ((current_gesture == "fist" or keyboard_place_action) and 
            hasattr(hit_info, 'entity') and 
            current_time - last_action_time > action_cooldown):
            
            new_position = hit_info.entity.position + hit_info.normal
            new_pos_tuple = (round(new_position.x), round(new_position.y), round(new_position.z))
            
            if new_position.y > 0:
                new_block = Voxel(position=new_pos_tuple)
                print(f"Placed block at {new_pos_tuple}")
                last_action_time = current_time
                keyboard_place_action = False
        
        # Block destruction
        elif ((current_gesture == "peace" or keyboard_destroy_action) and 
              hasattr(hit_info, 'entity') and 
              current_time - last_action_time > action_cooldown):
            
            entity_to_destroy = hit_info.entity
            if (isinstance(entity_to_destroy, Voxel) and 
                not getattr(entity_to_destroy, 'is_floor', False)):
                print(f"Destroyed block at {entity_to_destroy.position}")
                destroy(entity_to_destroy)
                last_action_time = current_time
                keyboard_destroy_action = False
    else:
        cursor.position = ray_origin + ray_direction * 5
        cursor.visible = True

def input(key):
    if key == 'escape':
        global running
        running = False
        application.quit()
    # Add debug key
    elif key == 't':
        print("Debug - Player position:", player.position)
        print("Debug - Camera position:", camera.position)
        print("Debug - Test cubes should be visible")

# Mouse control
def mouse_moved():
    if held_keys['left mouse']:
        global camera_rotation_x, camera_rotation_y
        camera_rotation_y += mouse.velocity[0] * 50
        camera_rotation_x -= mouse.velocity[1] * 50
        camera_rotation_x = max(-90, min(90, camera_rotation_x))

mouse.update = mouse_moved

# Cleanup
def cleanup():
    global running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()

import atexit
atexit.register(cleanup)

# Simplified HUD
hud = Text(
    text="""MINECRAFT VR - GRAPHICS FIXED

KEYBOARD: WASD + Arrows + Space + X + ESC
HAND GESTURES: Same as before

Should see: CYAN background + 3 colored cubes + white text

Press 'T' for debug info""",
    origin=(-.5, .5),
    scale=1,
    x=-.85,
    y=.45,
    background=True
)

print("\n" + "="*60)
print("GRAPHICS FIX APPLIED!")
print("You should now see:")
print("1. CYAN (blue) background instead of grey")
print("2. 3 bright colored test cubes")  
print("3. White 3D text saying 'MINECRAFT VR WORKING!'")
print("4. Brown floor blocks")
print("")
print("If still grey, your friend needs to update graphics drivers")
print("="*60)

app.run()
