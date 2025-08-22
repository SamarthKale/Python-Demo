from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import cv2
import mediapipe as mp
import math
import threading
import time
from collections import deque
import sys
import os

print("Starting Minecraft VR (CPU-Only Version)...")
print("=== IMPORTANT ===")
print("Two windows will open:")
print("1. Hand tracking camera window")
print("2. 3D Minecraft game window")
print("If you only see the camera, check your taskbar for the game window!")
print("================")

# Force CPU-only mode for MediaPipe (compatible with your versions)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow warnings
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Fix Windows camera issues

# Create Ursina app with Python 3.10.11 + Ursina 5.1.0 compatibility
app = Ursina()

# Proper window configuration for your setup
window.title = "Minecraft VR - 3D Game"
window.borderless = False
window.exit_button.visible = False
window.fps_counter.enabled = True
window.vsync = 0  # Use integer instead of boolean for Ursina 5.1.0
window.show_ursina_splash = False
window.size = (1024, 768)  # Set explicit window size

# CRITICAL: Set up proper lighting system (fixes grey screen)
# Remove any existing lights first
for entity in [e for e in scene.entities if hasattr(e, 'light_type')]:
    destroy(entity)

# Add proper lighting
sun = DirectionalLight()
sun.position = (10, 10, 10)
sun.look_at((0, 0, 0))
sun.color = color.white
sun.shadows = False  # Disable shadows for better performance

# Add ambient lighting
ambient = AmbientLight()
ambient.color = color.rgba(150, 150, 150, 1)  # Brighter ambient light

# Set camera properties
camera.fov = 70  # Slightly narrower FOV for better visibility
camera.clip_plane_near = 0.1
camera.clip_plane_far = 1000

# Create sky with proper texture fallback
try:
    Sky(color=color.rgb(135, 206, 235))  # Sky blue fallback
except:
    # If sky texture fails, create a simple colored background
    Entity(model='cube', color=color.rgb(135, 206, 235), scale=999, z=500)

print("✓ Ursina app created successfully")
print("✓ Lighting and sky setup complete")

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

    def input(self, key):
        if self.hovered:
            if key == 'right mouse down':
                # Place block on the face that was clicked
                voxel = Voxel(position=self.position + mouse.normal)
            if key == 'left mouse down' and not self.is_floor:
                # Destroy block (but not floor)
                destroy(self)

print("Creating floor...")
# Create smaller floor for better performance (20x20)
floor_blocks = []
for z in range(20):
    for x in range(20):
        voxel = Voxel(position=(x, 0, z), is_floor=True)
        floor_blocks.append(voxel)

# Place player in the center of the field
player = FirstPersonController()
player.position = (10, 1, 10)  # Center of 20x20 field
player.cursor.visible = False  # Hide default cursor

# Disable default mouse controls to implement our own
player.mouse_sensitivity = Vec2(0, 0)

# Custom camera rotation variables
camera_rotation_x = 0  # Vertical rotation (pitch)
camera_rotation_y = 0  # Horizontal rotation (yaw)
vertical_look_speed = 15  # Degrees per second for vertical look

# Floating cursor where finger points
cursor = Entity(model='sphere', color=color.red, scale=0.2, always_on_top=True)

print("✓ Game world created")
print("✓ Floor and player setup complete")

# ---------- Enhanced Hand Tracking with CPU-only MediaPipe ----------
try:
    # Initialize MediaPipe with your specific version (0.10.11)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Optimized settings for MediaPipe 0.10.11 CPU performance
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,  # Higher confidence for stability
        min_tracking_confidence=0.5,
        model_complexity=0  # Simplest model for CPU
    )
    mp_available = True
    print("✓ MediaPipe 0.10.11 initialized in CPU mode")
except Exception as e:
    print(f"MediaPipe initialization failed: {e}")
    mp_available = False
    hands = None

# Enhanced camera setup for OpenCV 4.9.0.80/4.11.0.86
cap = None
camera_available = False

def init_camera():
    global cap, camera_available
    try:
        print("Initializing camera with OpenCV", cv2.__version__)
        
        # Try DirectShow backend first (Windows), then default
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        camera_indices = [0, 1, 2]
        
        for backend in backends:
            for camera_index in camera_indices:
                try:
                    cap = cv2.VideoCapture(camera_index, backend)
                    
                    if cap.isOpened():
                        # Configure camera properties for your OpenCV version
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                        
                        # Test frame capture
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            camera_available = True
                            print(f"✓ Camera {camera_index} ready (backend: {backend})")
                            return True
                        else:
                            cap.release()
                            cap = None
                except Exception as e:
                    if cap:
                        cap.release()
                        cap = None
                    continue
        
        print("⚠ No working camera found - keyboard mode only")
        return False
        
    except Exception as e:
        print(f"Camera setup error: {e}")
        camera_available = False
        return False

# Initialize camera
init_camera()

# Shared variables with thread safety
gesture_lock = threading.Lock()
right_hand_gesture = None
left_hand_gesture = None
aim_x, aim_y = 0, 0
running = True

# Action cooldown
action_cooldown = 0.3  # Slightly longer cooldown for stability
last_action_time = 0

def get_hand_gesture(landmarks):
    """Improved gesture detection for CPU efficiency"""
    try:
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
        
        # Gesture classification with better reliability
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
    except Exception:
        return "neutral"

def hand_tracker():
    """Hand tracker optimized for your exact setup"""
    global right_hand_gesture, left_hand_gesture, aim_x, aim_y, running
    
    if not camera_available or not mp_available:
        print("Hand tracking not available - keyboard mode active")
        return
    
    frame_count = 0
    
    # Create optimized window for your system
    window_name = "Minecraft VR - Hand Control"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 50, 50)
    
    print("✓ Hand tracking active - CPU optimized mode")
    
    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, retrying...")
                time.sleep(0.1)
                continue
            
            # Process every 2nd frame for CPU efficiency
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            
            # Flip for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(rgb_frame)
            
            h, w, _ = frame.shape
            current_right_gesture = None
            current_left_gesture = None
            
            # Enhanced interface zones with better visibility
            cv2.rectangle(frame, (0, 0), (w//2, h), (255, 100, 100), 4)  # Left zone (red-ish)
            cv2.rectangle(frame, (w//2, 0), (w, h), (100, 255, 100), 4)  # Right zone (green-ish)
            
            # Larger, clearer labels
            cv2.putText(frame, "LEFT HAND MOVEMENT", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            cv2.putText(frame, "RIGHT HAND BUILDING", (w//2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
            
            # Enhanced status indicator with text
            status_color = (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255)
            status_text = "HANDS DETECTED" if results.multi_hand_landmarks else "NO HANDS"
            cv2.circle(frame, (w-50, 30), 15, status_color, -1)
            cv2.putText(frame, status_text, (w-200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmarks = hand_landmarks.landmark
                    
                    # Determine which hand (note: labels are flipped due to mirror)
                    hand_label = handedness.classification[0].label
                    is_right_hand = (hand_label == "Left")  # Flipped due to mirror
                    
                    gesture = get_hand_gesture(landmarks)
                    
                    if is_right_hand:  # Right hand controls building
                        current_right_gesture = gesture
                        
                        # Get index finger position for aiming
                        index_tip = landmarks[8]
                        with gesture_lock:
                            aim_x = (index_tip.x - 0.5) * 1.5  # Reduced sensitivity
                            aim_y = (0.5 - index_tip.y) * 1.5
                        
                        # Visual feedback with better colors and sizes
                        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                        
                        if gesture == "fist":
                            cv2.circle(frame, (ix, iy), 25, (0, 255, 0), -1)  # Larger green circle
                            cv2.putText(frame, "BUILD BLOCK", (w//2 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        elif gesture == "peace":
                            cv2.circle(frame, (ix, iy), 25, (0, 0, 255), -1)  # Larger red circle
                            cv2.putText(frame, "DESTROY BLOCK", (w//2 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        elif gesture == "point":
                            cv2.circle(frame, (ix, iy), 18, (255, 255, 0), -1)  # Yellow for aim
                            cv2.putText(frame, "AIMING", (w//2 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        else:
                            cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)
                            cv2.putText(frame, "Ready to Build", (w//2 + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    else:  # Left hand controls movement
                        current_left_gesture = gesture
                        
                        # Get palm center for movement
                        palm_center = landmarks[9]
                        px, py = int(palm_center.x * w), int(palm_center.y * h)
                        
                        if gesture == "fist":
                            cv2.circle(frame, (px, py), 25, (0, 255, 255), -1)  # Cyan for forward
                            cv2.putText(frame, "MOVE FORWARD", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        elif gesture == "open":
                            cv2.circle(frame, (px, py), 25, (255, 0, 255), -1)  # Magenta for back
                            cv2.putText(frame, "MOVE BACKWARD", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        elif gesture == "point":
                            cv2.circle(frame, (px, py), 25, (255, 165, 0), -1)  # Orange for turn
                            cv2.putText(frame, "TURN LEFT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                        elif gesture == "peace":
                            cv2.circle(frame, (px, py), 25, (0, 165, 255), -1)  # Blue for look down
                            cv2.putText(frame, "LOOK DOWN", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        elif gesture == "three":
                            cv2.circle(frame, (px, py), 25, (0, 255, 165), -1)  # Green for look up
                            cv2.putText(frame, "LOOK UP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 165), 2)
                        else:
                            cv2.circle(frame, (px, py), 15, (255, 255, 255), 2)
                            cv2.putText(frame, "Ready to Move", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Draw hand landmarks (simplified for performance)
                    if mp_available:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Update shared variables thread-safely
            with gesture_lock:
                right_hand_gesture = current_right_gesture
                left_hand_gesture = current_left_gesture
            
            # Add comprehensive performance and instruction info
            instruction_y = h - 80
            cv2.putText(frame, "GESTURES: Fist=Forward | Open=Back | Point=Left | Peace=Down | Three=Up", 
                       (5, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "BUILDING: Point=Aim | Fist=Build | Peace=Destroy", 
                       (5, instruction_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Frame: {frame_count} | Movement: FIXED | Speed: SLOW", 
                       (5, instruction_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, "ESC to quit | Movement system optimized", 
                       (5, instruction_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Hand tracking stopped by user")
                running = False
                break
                
        except Exception as e:
            print(f"Hand tracking error: {e}")
            time.sleep(0.1)
            continue
    
    # Cleanup
    print("Cleaning up hand tracking...")
    try:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except:
        pass

# Start hand tracking thread if available
if camera_available and mp_available:
    print("Starting optimized hand tracking thread...")
    threading.Thread(target=hand_tracker, daemon=True).start()
else:
    print("Hand tracking disabled - keyboard controls only")

# ---------- Movement and Camera Control ----------
movement_speed = 2.5  # Slower, more controlled movement
rotation_speed = 30   # Slower rotation for better control
vertical_look_speed = 20  # Slower vertical look

def handle_controls():
    """Handle both keyboard and gesture controls with proper movement"""
    global left_hand_gesture, camera_rotation_x, camera_rotation_y
    
    # Get current hand gesture safely
    with gesture_lock:
        current_left_gesture = left_hand_gesture
    
    # Hand gesture movement controls - FIXED MOVEMENT SYSTEM
    if current_left_gesture == "fist":
        # Move forward in the direction player is facing
        forward_dir = Vec3(
            math.sin(math.radians(player.rotation_y)),
            0,
            math.cos(math.radians(player.rotation_y))
        ).normalized()
        player.position += forward_dir * movement_speed * time.dt
        
    elif current_left_gesture == "open":
        # Move backward opposite to facing direction  
        forward_dir = Vec3(
            math.sin(math.radians(player.rotation_y)),
            0,
            math.cos(math.radians(player.rotation_y))
        ).normalized()
        player.position -= forward_dir * movement_speed * time.dt
        
    elif current_left_gesture == "point":
        # Turn left (rotate camera)
        camera_rotation_y -= rotation_speed * time.dt
        
    elif current_left_gesture == "three":
        # Look up
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
        
    elif current_left_gesture == "peace":
        # Look down  
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Keyboard movement controls (WASD) - FIXED MOVEMENT
    if held_keys['w']:
        # Forward movement
        forward_dir = Vec3(
            math.sin(math.radians(player.rotation_y)),
            0,
            math.cos(math.radians(player.rotation_y))
        ).normalized()
        player.position += forward_dir * movement_speed * time.dt
        
    if held_keys['s']:
        # Backward movement
        forward_dir = Vec3(
            math.sin(math.radians(player.rotation_y)),
            0,
            math.cos(math.radians(player.rotation_y))
        ).normalized()
        player.position -= forward_dir * movement_speed * time.dt
        
    if held_keys['a']:
        # Strafe left
        right_dir = Vec3(
            math.cos(math.radians(player.rotation_y)),
            0,
            -math.sin(math.radians(player.rotation_y))
        ).normalized()
        player.position -= right_dir * movement_speed * time.dt
        
    if held_keys['d']:
        # Strafe right
        right_dir = Vec3(
            math.cos(math.radians(player.rotation_y)),
            0,
            -math.sin(math.radians(player.rotation_y))
        ).normalized()
        player.position += right_dir * movement_speed * time.dt
    
    # Keyboard look controls (Arrow Keys)
    if held_keys['left arrow']:
        camera_rotation_y -= rotation_speed * time.dt
    if held_keys['right arrow']:
        camera_rotation_y += rotation_speed * time.dt
    if held_keys['up arrow']:
        camera_rotation_x = max(camera_rotation_x - vertical_look_speed * time.dt, -90)
    if held_keys['down arrow']:
        camera_rotation_x = min(camera_rotation_x + vertical_look_speed * time.dt, 90)
    
    # Mouse look (hold right mouse button)
    if held_keys['right mouse']:
        camera_rotation_y += mouse.velocity[0] * 25  # Slower mouse sensitivity
        camera_rotation_x -= mouse.velocity[1] * 25
        camera_rotation_x = max(-90, min(90, camera_rotation_x))
    
    # Apply rotations to player and camera
    player.rotation_y = camera_rotation_y
    camera.rotation_x = camera_rotation_x

# ---------- Block System ----------
def update():
    """Main game update loop"""
    global right_hand_gesture, last_action_time
    
    # Handle movement and camera
    handle_controls()
    
    current_time = time.time()
    
    # Get current gesture safely
    with gesture_lock:
        current_gesture = right_hand_gesture
    
    # Create aiming ray
    ray_origin = camera.world_position
    fov_rad = math.radians(camera.fov)
    aspect = window.aspect_ratio
    
    # Use hand aiming if available, otherwise use screen center
    target_x = aim_x if current_gesture else 0
    target_y = aim_y if current_gesture else 0
    
    ray_direction = Vec3(
        target_x * math.tan(fov_rad / 2) * aspect,
        target_y * math.tan(fov_rad / 2),
        1
    ).normalized()
    
    ray_direction = camera.right * ray_direction.x + camera.up * ray_direction.y + camera.forward
    ray_direction = ray_direction.normalized()
    
    # Perform raycast
    hit_info = raycast(ray_origin, ray_direction, distance=15, ignore=[player, cursor])
    
    if hit_info.hit:
        cursor.position = hit_info.world_point
        cursor.visible = True
        
        # Block placement (gesture or keyboard)
        if ((current_gesture == "fist" or held_keys['space']) and 
            hasattr(hit_info, 'entity') and 
            current_time - last_action_time > action_cooldown):
            
            new_position = hit_info.entity.position + hit_info.normal
            new_pos_tuple = (round(new_position.x), round(new_position.y), round(new_position.z))
            
            # Check if position is free
            existing = False
            for entity in scene.entities:
                if (isinstance(entity, Voxel) and 
                    distance(entity.position, new_position) < 0.5):
                    existing = True
                    break
            
            if not existing and new_position.y > 0:
                Voxel(position=new_pos_tuple)
                last_action_time = current_time
        
        # Block destruction (gesture or keyboard)
        elif ((current_gesture == "peace" or held_keys['x']) and 
              hasattr(hit_info, 'entity') and 
              current_time - last_action_time > action_cooldown):
            
            entity_to_destroy = hit_info.entity
            if (isinstance(entity_to_destroy, Voxel) and 
                not getattr(entity_to_destroy, 'is_floor', False)):
                destroy(entity_to_destroy)
                last_action_time = current_time
    else:
        cursor.position = ray_origin + ray_direction * 10
        cursor.visible = True

def input(key):
    """Handle key input events"""
    if key == 'escape':
        global running
        running = False
        print("Shutting down...")
        sys.exit()

# Instructions HUD - Enhanced and clearer
instructions = Text(
    text="""MINECRAFT VR - COMPLETE CONTROLS

KEYBOARD CONTROLS:
W/S - Move Forward/Backward
A/D - Strafe Left/Right  
↑↓←→ - Look Up/Down/Left/Right
Right Mouse + Drag - Mouse Look
SPACE - Place Block
X - Destroy Block
ESC - Quit Game

HAND GESTURES:
RIGHT HAND (Building):
  👉 Point Finger = Aim Cursor
  ✊ Fist = Build Block
  ✌️ Peace Sign = Destroy Block

LEFT HAND (Movement):
  ✊ Fist = Move Forward
  ✋ Open Palm = Move Backward  
  👉 Point = Turn Left
  ✌️ Peace Sign = Look Down
  🤟 Three Fingers = Look Up

TIPS:
• Movement is slower for better control
• All original functions preserved
• Use keyboard if hand tracking fails""",
    origin=(-.5, .5),
    scale=0.6,
    x=-.95,
    y=.45,
    background=True,
    color=color.white
)

# Enhanced system info with movement status
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
system_info = Text(
    text=f"Python {python_version} | OpenCV {cv2.__version__} | Ursina 5.1.0\n" + 
         f"Movement: FIXED | Speed: SLOW | Camera: {'✓' if camera_available else '✗'} | MediaPipe: {'✓' if mp_available else '✗'}",
    origin=(-.5, -.5),
    scale=0.55,
    x=-.95,
    y=-.45,
    background=True,
    color=color.cyan
)

# Enhanced startup message
if camera_available and mp_available:
    status_msg = "✓ Hand tracking ready - Movement FIXED"
    color_status = color.lime
else:
    status_msg = "⚠ Keyboard controls only - Movement FIXED"
    color_status = color.orange

startup_text = Text(
    text=f"MINECRAFT VR READY!\nMovement System FIXED\n{status_msg}\nUse slow, controlled movements",
    origin=(0, 0),
    scale=1.1,
    color=color_status,
    background=True
)

def hide_startup():
    destroy(startup_text)

invoke(hide_startup, delay=4)

# Cleanup function
def cleanup():
    global running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup completed")

import atexit
atexit.register(cleanup)

print("✓ All systems initialized!")
print("✓ Starting game loop...")
print(f"✓ Camera: {'Available' if camera_available else 'Not available'}")
print(f"✓ MediaPipe: {'CPU-only mode' if mp_available else 'Not available'}")
print("\n" + "="*60)
print("GAME READY! Check for both windows:")
print("1. This 3D game window")
if camera_available:
    print("2. Hand tracking camera window")
else:
    print("2. No camera - keyboard controls only")
print("="*60)

# Run the application
app.run()