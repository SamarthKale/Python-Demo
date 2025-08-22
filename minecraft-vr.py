from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import cv2
import mediapipe as mp
import math
import threading
import time
from collections import deque

print("Starting Minecraft VR...")
print("=== IMPORTANT ===")
print("Two windows will open:")
print("1. Hand tracking camera window")
print("2. 3D Minecraft game window")
print("If you only see the camera, check your taskbar for the game window!")
print("================")

# Force Ursina window to appear on top and be visible
app = Ursina()
window.title = "Minecraft VR - 3D Game"
window.borderless = False
window.exit_button.visible = False
window.fps_counter.enabled = True

# Make window more visible
camera.fov = 90

print("✓ Ursina app created successfully")
print("✓ 3D window should be visible now")

# ---------- Game Blocks ----------
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

print("Creating floor...")
# Create much larger floor (30x30)
floor_blocks = []
for z in range(30):
    for x in range(30):
        voxel = Voxel(position=(x, 0, z), is_floor=True)
        floor_blocks.append(voxel)

# Place player in the center of the field
player = FirstPersonController()
player.position = (15, 1, 15)  # Center of 30x30 field

# Disable default mouse controls to implement our own
player.mouse_sensitivity = Vec2(0, 0)

# Custom camera rotation variables
camera_rotation_x = 0  # Vertical rotation (pitch)
camera_rotation_y = 0  # Horizontal rotation (yaw)
vertical_look_speed = 15  # Degrees per second for vertical look

# Floating cursor where finger points
cursor = Entity(model='sphere', color=color.azure, scale=0.1, always_on_top=True)

print("✓ Game world created")
print("✓ Floor and player setup complete")

# ---------- Enhanced Hand Tracking ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Fixed camera setup with proper settings
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera")
        cap = None
    else:
        # Basic resolution and FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Fix auto-exposure and color issues
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_CONTRAST, 128)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
        cap.set(cv2.CAP_PROP_HUE, 0)
        cap.set(cv2.CAP_PROP_GAIN, 0)
        
        print("✓ Camera initialized successfully")
except Exception as e:
    print(f"Camera error: {e}")
    cap = None

# Shared variables with thread safety
gesture_lock = threading.Lock()
right_hand_gesture = None
left_hand_gesture = None
aim_x, aim_y = 0, 0
running = True

# Action cooldown
action_cooldown = 0.1
last_action_time = 0

def get_hand_gesture(landmarks):
    """Simplified gesture detection with easy gestures"""
    fingers = []
    
    # Thumb detection (compare x coordinates)
    if landmarks[4].x > landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers (compare y coordinates)
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    total_fingers = sum(fingers)
    
    # Simplified gesture classification
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
    global right_hand_gesture, left_hand_gesture, aim_x, aim_y, last_action_time, running
    
    if not cap:
        print("No camera available - hand tracking disabled")
        print("You can still use keyboard controls!")
        return
    
    frame_count = 0
    
    # Set window position to not overlap with game window
    cv2.namedWindow("Hand Control - Natural View", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Hand Control - Natural View", 100, 100)
    
    print("✓ Hand tracking window created")
    print("✓ Camera should be showing now")
    
    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Skip frames for better performance
            frame_count += 1
            if frame_count % 2 == 0:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            h, w, _ = frame.shape
            current_time = time.time()
            current_right_gesture = None
            current_left_gesture = None
            
            # Enhanced visual zones
            cv2.rectangle(frame, (0, 0), (w//2, h), (100, 100, 255), 3)
            cv2.rectangle(frame, (w//2, 0), (w, h), (255, 100, 100), 3)
            cv2.putText(frame, "LEFT HAND ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            cv2.putText(frame, "RIGHT HAND ZONE", (w//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            
            # Add status text
            cv2.putText(frame, "3D Game Window should be open!", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
                        
                        # Visual feedback for right hand
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
                    
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Update shared variables
            with gesture_lock:
                right_hand_gesture = current_right_gesture
                left_hand_gesture = current_left_gesture
            
            # Display frame
            cv2.imshow("Hand Control - Natural View", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                running = False
                break
                
        except Exception as e:
            print(f"Hand tracking error: {e}")
            time.sleep(0.1)
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()

# Start hand tracking in separate thread
if cap:
    print("Starting hand tracking thread...")
    threading.Thread(target=hand_tracker, daemon=True).start()

# ---------- Movement and Camera Control ----------
movement_speed = 1.5
rotation_speed = 30

def handle_controls():
    global left_hand_gesture, camera_rotation_x, camera_rotation_y, last_action_time
    
    with gesture_lock:
        current_left_gesture = left_hand_gesture
    
    # Hand gesture movement controls
    if current_left_gesture == "fist":
        player.position += player.forward * movement_speed * time.dt
    elif current_left_gesture == "open":
        player.position -= player.forward * movement_speed * time.dt
    
    # Hand gesture look controls
    if current_left_gesture == "point":
        camera_rotation_y -= rotation_speed * time.dt
    elif current_left_gesture == "three":
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
    elif current_left_gesture == "peace":
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Keyboard movement controls (WASD)
    if held_keys['w']:
        player.position += player.forward * movement_speed * time.dt
    if held_keys['s']:
        player.position -= player.forward * movement_speed * time.dt
    if held_keys['a']:
        player.position -= player.right * movement_speed * time.dt
    if held_keys['d']:
        player.position += player.right * movement_speed * time.dt
    
    # Keyboard look controls (Arrow Keys)
    if held_keys['left arrow']:
        camera_rotation_y -= rotation_speed * time.dt
    if held_keys['right arrow']:
        camera_rotation_y += rotation_speed * time.dt
    if held_keys['up arrow']:
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
    if held_keys['down arrow']:
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Apply rotations to player
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
    
    # Check for keyboard actions
    if held_keys['space']:
        keyboard_place_action = True
    if held_keys['x']:
        keyboard_destroy_action = True
    
    # Create ray for aiming
    ray_origin = camera.world_position
    fov_rad = math.radians(camera.fov)
    aspect = window.aspect_ratio
    
    ray_direction = Vec3(
        aim_x * math.tan(fov_rad / 2) * aspect,
        aim_y * math.tan(fov_rad / 2),
        1
    ).normalized()
    
    ray_direction = camera.right * ray_direction.x + camera.up * ray_direction.y + camera.forward
    ray_direction = ray_direction.normalized()
    
    # Raycast
    hit_info = raycast(ray_origin, ray_direction, distance=20, ignore=[player, cursor])
    
    if hit_info.hit:
        cursor.position = hit_info.world_point
        cursor.visible = True
        
        # Block placement
        if ((current_gesture == "fist" or keyboard_place_action) and 
            hasattr(hit_info, 'entity') and 
            current_time - last_action_time > action_cooldown):
            
            new_position = hit_info.entity.position + hit_info.normal
            new_pos_tuple = (round(new_position.x), round(new_position.y), round(new_position.z))
            
            existing = False
            for entity in scene.entities:
                if (isinstance(entity, Voxel) and 
                    distance(entity.position, new_position) < 0.5):
                    existing = True
                    break
            
            if not existing and new_position.y > 0:
                Voxel(position=new_pos_tuple)
                last_action_time = current_time
                keyboard_place_action = False
        
        # Block destruction
        elif ((current_gesture == "peace" or keyboard_destroy_action) and 
              hasattr(hit_info, 'entity') and 
              current_time - last_action_time > action_cooldown):
            
            entity_to_destroy = hit_info.entity
            if (isinstance(entity_to_destroy, Voxel) and 
                not getattr(entity_to_destroy, 'is_floor', False)):
                destroy(entity_to_destroy)
                last_action_time = current_time
                keyboard_destroy_action = False
    else:
        cursor.position = ray_origin + ray_direction * 10
        cursor.visible = True

def input(key):
    if key == 'escape':
        global running
        running = False
        application.quit()

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

# HUD Instructions
instructions_text = """MINECRAFT VR CONTROLS

KEYBOARD:
WASD - Move
Arrows - Look
Space - Place block
X - Destroy block
ESC - Quit

HAND GESTURES:
Right Hand (Building):
- Point = Aim
- Fist = Place block  
- Peace sign = Destroy block

Left Hand (Movement):
- Fist = Forward
- Open palm = Backward
- Point = Turn left
- 3 fingers = Look up
- Peace sign = Look down

Check taskbar if game window is hidden!
"""

hud = Text(
    text=instructions_text,
    origin=(-.5, .5),
    scale=0.8,
    x=-.85,
    y=.45,
    background=True
)

# Add visible startup message
startup_msg = Text(
    text="MINECRAFT VR LOADED!\nLook for the camera window and this 3D window!\nBoth should be visible now.",
    origin=(0, 0),
    scale=2,
    color=color.yellow,
    background=True
)

def remove_startup_msg():
    destroy(startup_msg)

# Remove startup message after 5 seconds
invoke(remove_startup_msg, delay=5)

print("✓ All systems ready!")
print("✓ Starting main game loop...")
print("\n" + "="*50)
print("IMPORTANT: Two windows should now be open:")
print("1. This 3D game window (Minecraft VR)")  
print("2. Hand tracking camera window")
print("If you only see one, check your taskbar!")
print("="*50)

app.run()
