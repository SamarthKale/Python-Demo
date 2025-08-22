import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe
face_mesh = mp_face.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Visual effects variables
time_start = time.time()
hand_trails = []
face_particles = []
energy_rings = []
fps_counter = deque(maxlen=30)

class Particle:
    def __init__(self, x, y, vx=0, vy=0, life=60, color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.vx = vx + np.random.uniform(-2, 2)
        self.vy = vy + np.random.uniform(-3, -1)
        self.life = life
        self.max_life = life
        self.color = color
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= 1
        
    def draw(self, frame):
        if self.life > 0:
            alpha = self.life / self.max_life
            size = int(alpha * 8)
            if size > 0:
                # Create glowing effect
                for i in range(3):
                    cv2.circle(frame, (int(self.x), int(self.y)), 
                             size + i*2, 
                             tuple(int(c * alpha * (0.3 + i*0.2)) for c in self.color), 
                             -1)

def create_neon_effect(frame, points, color, thickness=3):
    """Create a neon glow effect for lines"""
    if len(points) < 2:
        return
    
    # Draw outer glow
    for i in range(len(points)-1):
        for glow in range(5, 0, -1):
            alpha = 0.1 + (5-glow) * 0.15
            glow_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, points[i], points[i+1], glow_color, thickness + glow*2)
    
    # Draw bright inner line
    for i in range(len(points)-1):
        cv2.line(frame, points[i], points[i+1], color, thickness)

def draw_cyberpunk_face_outline(frame, landmarks):
    """Draw a cyberpunk-style face outline with glowing effects"""
    h, w = frame.shape[:2]
    
    # Face contour points
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    if len(landmarks.landmark) > max(face_oval):
        points = []
        for idx in face_oval:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            points.append((x, y))
        
        # Close the loop
        points.append(points[0])
        
        # Create animated color
        time_factor = time.time() * 3
        r = int(128 + 127 * math.sin(time_factor))
        g = int(128 + 127 * math.sin(time_factor + 2))
        b = int(128 + 127 * math.sin(time_factor + 4))
        
        create_neon_effect(frame, points, (b, g, r), 2)

def draw_energy_eyes(frame, landmarks):
    """Draw glowing energy effects around eyes"""
    h, w = frame.shape[:2]
    
    # Left and right eye indices
    left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    time_factor = time.time() * 5
    
    for eye_indices, offset in [(left_eye, 0), (right_eye, math.pi)]:
        if len(landmarks.landmark) > max(eye_indices):
            # Get eye center
            eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) 
                         for i in eye_indices]
            
            if eye_points:
                cx = sum(p[0] for p in eye_points) // len(eye_points)
                cy = sum(p[1] for p in eye_points) // len(eye_points)
                
                # Draw pulsing energy rings
                for ring in range(3):
                    radius = 20 + ring * 15 + int(10 * math.sin(time_factor + offset))
                    intensity = 255 - ring * 60
                    color = (intensity//3, intensity//2, intensity)
                    cv2.circle(frame, (cx, cy), radius, color, 2)

def draw_hand_energy_trail(frame, hand_landmarks, hand_trails, hand_index):
    """Create energy trails following hand movements"""
    h, w = frame.shape[:2]
    
    # Get index finger tip (landmark 8)
    tip_x = int(hand_landmarks.landmark[8].x * w)
    tip_y = int(hand_landmarks.landmark[8].y * h)
    
    # Ensure we have enough trail lists
    while len(hand_trails) <= hand_index:
        hand_trails.append(deque(maxlen=20))
    
    # Add current position to trail
    hand_trails[hand_index].appendleft((tip_x, tip_y))
    
    # Draw trail with fading effect
    if len(hand_trails[hand_index]) > 1:
        trail_points = list(hand_trails[hand_index])
        colors = [(255, 100, 255), (100, 255, 255)]  # Magenta to Cyan
        color = colors[hand_index % 2]
        
        create_neon_effect(frame, trail_points, color, 3)
    
    # Create particles at fingertip
    if np.random.random() > 0.7:  # 30% chance each frame
        face_particles.append(Particle(tip_x, tip_y, 
                                     vx=np.random.uniform(-3, 3),
                                     vy=np.random.uniform(-3, 3),
                                     life=40,
                                     color=(255, 150, 0)))

def draw_holographic_grid(frame):
    """Draw an animated holographic grid background"""
    h, w = frame.shape[:2]
    time_factor = time.time() * 2
    
    # Draw grid lines with animation
    grid_size = 50
    offset = int(time_factor * 20) % grid_size
    
    for i in range(-offset, w + grid_size, grid_size):
        alpha = 0.3 + 0.2 * math.sin(time_factor + i * 0.01)
        color = (int(50 * alpha), int(100 * alpha), int(150 * alpha))
        cv2.line(frame, (i, 0), (i, h), color, 1)
    
    for i in range(-offset, h + grid_size, grid_size):
        alpha = 0.3 + 0.2 * math.sin(time_factor + i * 0.01)
        color = (int(50 * alpha), int(100 * alpha), int(150 * alpha))
        cv2.line(frame, (0, i), (w, i), color, 1)

def draw_fps_and_info(frame):
    """Draw FPS counter and info with styling"""
    current_time = time.time()
    fps_counter.append(current_time)
    
    if len(fps_counter) > 1:
        fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "ESC: Exit | SPACE: Toggle Effects", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# Main loop
effects_enabled = True
print("🚀 Advanced Face & Hand Tracker Started!")
print("Controls:")
print("  ESC - Exit")
print("  SPACE - Toggle effects")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Create dark background for better effect visibility
    frame = cv2.addWeighted(frame, 0.6, np.zeros_like(frame), 0.4, 0)
    
    if effects_enabled:
        # Draw holographic grid background
        draw_holographic_grid(frame)
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process face
    face_result = face_mesh.process(rgb)
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            if effects_enabled:
                draw_cyberpunk_face_outline(frame, face_landmarks)
                draw_energy_eyes(frame, face_landmarks)
            else:
                # Basic face mesh
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
    
    # Process hands
    hand_result = hands.process(rgb)
    if hand_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
            if effects_enabled:
                draw_hand_energy_trail(frame, hand_landmarks, hand_trails, idx)
                
                # Draw hand skeleton with neon effect
                connections = list(mp_hands.HAND_CONNECTIONS)
                hand_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_points) and end_idx < len(hand_points):
                        create_neon_effect(frame, [hand_points[start_idx], hand_points[end_idx]], 
                                         (255, 255, 100), 2)
            else:
                # Basic hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    
    # Update and draw particles
    if effects_enabled:
        for particle in face_particles[:]:
            particle.update()
            if particle.life <= 0:
                face_particles.remove(particle)
            else:
                particle.draw(frame)
    
    # Draw UI
    frame = draw_fps_and_info(frame)
    
    cv2.imshow("🔥 Advanced Face & Hand Tracker", frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        effects_enabled = not effects_enabled
        print(f"Effects {'enabled' if effects_enabled else 'disabled'}")

cap.release()
cv2.destroyAllWindows()
print("👋 Tracker closed!")
