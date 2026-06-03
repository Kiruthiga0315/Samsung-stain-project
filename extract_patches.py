import os
import glob
import cv2
import numpy as np
import openslide

# Setup directory paths
SLIDE_DIR = 'data/raw_slides/'
UNSTAINED_DIR = 'data/he_paired/unstained'
STAINED_DIR = 'data/he_paired/stained'
OUT_DIR = 'data/he_train'

os.makedirs(UNSTAINED_DIR, exist_ok=True)
os.makedirs(STAINED_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 256
TARGET_PATCHES = 550  # Gives a comfortable safety margin over 500
patch_count = 0

# Find all downloaded .svs files recursively
slide_paths = glob.glob(os.path.join(SLIDE_DIR, '**/*.svs'), recursive=True)

if not slide_paths:
    print("No .svs files found. Please check your data/raw_slides/ folder.")
    exit()

print(f"Found {len(slide_paths)} slides. Starting patch extraction...")

for slide_path in slide_paths:
    if patch_count >= TARGET_PATCHES:
        break
        
    try:
        slide = openslide.OpenSlide(slide_path)
        width, height = slide.dimensions
        
        # Grid step: Use a stride of PATCH_SIZE * 3 to sample across different tissue areas
        for y in range(0, height, PATCH_SIZE * 3):
            for x in range(0, width, PATCH_SIZE * 3):
                if patch_count >= TARGET_PATCHES:
                    break
                
                # Extract patch at highest resolution (level 0)
                patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                patch_rgb = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2BGR)
                
                # Filter out empty background (pure white/bright space)
                gray_patch = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
                white_pixels = np.sum(gray_patch > 220)
                total_pixels = PATCH_SIZE * PATCH_SIZE
                
                # If the patch is more than 50% blank background, skip it
                if (white_pixels / total_pixels) > 0.5:
                    continue
                
                # 1. Ground Truth: Target H&E Stained patch
                stained_filename = f"patch_{patch_count}.png"
                cv2.imwrite(os.path.join(STAINED_DIR, stained_filename), patch_rgb)
                
                # 2. Input Simulation: Grayscale Unstained structural patch
                unstained_patch = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
                unstained_patch = cv2.cvtColor(unstained_patch, cv2.COLOR_GRAY2BGR) # Keep 3 channels
                cv2.imwrite(os.path.join(UNSTAINED_DIR, stained_filename), unstained_patch)
                
                # 3. Concatenated image for standard pix2pix input (512x256 side-by-side)
                # Left side: Input (Unstained) | Right side: Target (Stained)
                pair = np.hstack([unstained_patch, patch_rgb])
                cv2.imwrite(os.path.join(OUT_DIR, stained_filename), pair)
                
                patch_count += 1
                if patch_count % 50 == 0:
                    print(f"Extracted {patch_count}/{TARGET_PATCHES} patches...")
                    
    except Exception as e:
        print(f"Skipping slide {os.path.basename(slide_path)} due to an error: {e}")

print(f"\nSuccess! Generated {patch_count} image pairs inside: {OUT_DIR}")