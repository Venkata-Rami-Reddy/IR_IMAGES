import numpy as np
import cv2
import os
import math

# --- 1. SETUP & CONSTANTS ---
FRAME_COUNT = 30
IMAGE_SIZE = 256
OUTPUT_DIR = "ir_frames"

# Temperature Constants (in Celsius)
T_MIN, T_MAX = 15.0, 40.0 # Total scene range
T_BG = 20.0               # Background temperature (Cooler)
T_OBJ = 35.0              # Object temperature (Warmer)

# Mapping Constants (8-bit grayscale)
I_MIN, I_MAX = 0, 255

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_temp_to_intensity(temp_map):
    """Maps a floating-point temperature array to 8-bit intensity (0-255)."""
    # Normalize temperature based on the full scene range
    norm_temp = (temp_map - T_MIN) / (T_MAX - T_MIN)
    # Scale to intensity range
    intensity = norm_temp * (I_MAX - I_MIN) + I_MIN
    # Clip and convert to unsigned 8-bit integer for saving
    return np.clip(intensity, I_MIN, I_MAX).astype(np.uint8)

# --- 2. BASE FRAME DEFINITION (Initial Object) ---
def create_base_frame(size):
    """Creates a background array and a simple object mask."""
    # Start with a uniform cool background
    temp_map = np.full((size, size), T_BG, dtype=np.float32)

    # Simple object shape: A warm circle (Simulating a human-like silhouette)
    center_x, center_y = size // 2, size // 2
    radius = size // 8

    # Create coordinate grid
    y, x = np.ogrid[-center_y:size - center_y, -center_x:size - center_x]
    
    # Calculate distance from center
    mask = x*x + y*y <= radius*radius
    
    return temp_map, mask.astype(np.float32)

# --- 3. TEMPORAL LOOP & MOTION ---
def generate_frame(frame_num, total_frames):
    """Generates a single frame with motion and effects."""
    temp_map, object_mask = create_base_frame(IMAGE_SIZE)
    
    # Define simple motion: diagonal movement from top-left to bottom-right
    # Uses a simple linear interpolation for position (x, y)
    max_shift = IMAGE_SIZE // 3
    
    # Calculate the shift for the current frame
    # f = current_frame_num / total_frames (0.0 to 1.0)
    f = frame_num / total_frames 
    shift_x = int(max_shift * f)
    shift_y = int(max_shift * f)

    # Shift the object mask using numpy roll (simulating movement)
    shifted_mask = np.roll(object_mask, (shift_y, shift_x), axis=(0, 1))

    # Apply the object temperature to the map
    temp_map[shifted_mask == 1] = T_OBJ
    
    # Convert the temperature map to intensity (pre-effects image)
    img_data = map_temp_to_intensity(temp_map)

    # --- 4. SENSOR EFFECTS ---
    
    # Effect 1: Gaussian Noise (Simulating FPN/Electronic Noise)
    # Scale=5 is a subtle noise level
    noise_scale = 5.0 
    noise = np.random.normal(loc=0, scale=noise_scale, size=img_data.shape)
    noisy_img = np.clip(img_data + noise, 0, 255).astype(np.uint8)
    
    # Effect 2: Gaussian Blur (Simulating Atmospheric Attenuation/Diffusion)
    # 3x3 kernel size, sigma=0
    final_img = cv2.GaussianBlur(noisy_img, (3, 3), 0) 
    
    # Save the frame
    filename = os.path.join(OUTPUT_DIR, f"frame_{frame_num:03d}.png")
    cv2.imwrite(filename, final_img)
    
    return filename

# Run the generation loop
print(f"Generating {FRAME_COUNT} IR frames...")
for i in range(1, FRAME_COUNT + 1):
    path = generate_frame(i, FRAME_COUNT)
    print(f"Generated: {path}")

print("Generation complete! The frames are in the 'ir_frames' folder.")