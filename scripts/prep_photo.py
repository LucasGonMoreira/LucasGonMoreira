"""
Prepare a portrait photo for clean ASCII conversion:
  1. remove the background (rembg) so the subject is isolated
  2. boost LOCAL contrast (CLAHE) so a flatly-lit face gains highlights and
     shadows -- this is what turns a dark blob into a recognizable face
  3. composite the subject onto pure white so the background reads as blank
     (white -> spaces in the ascii ramp)

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.
Run once whenever the source photo changes; the ascii SVG itself is static.

    python scripts/prep_photo.py <input.jpg> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import remove

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

# 1. cut out the subject. Keep RGB from the original image: rembg clears the
# color channels of transparent pixels, but we restore the phone below.
source = Image.open(INP).convert("RGBA")
cut = remove(source)
rgb = np.array(source.convert("RGB"))
alpha = np.array(cut.split()[-1])                 # 0 = background

# rembg recognizes the person and hand well, but can treat the dark phone as
# part of the background. Preserve the phone body for this portrait using a
# small polygon expressed as proportions of the image size, so it keeps
# working if the source photo is resized without changing its composition.
h, w = alpha.shape
phone_mask = np.zeros_like(alpha)
x1, y1 = int(w * 0.600), int(h * 0.285)
x2, y2 = int(w * 0.805), int(h * 0.605)
radius = max(1, int(w * 0.018))
cv2.rectangle(phone_mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
cv2.rectangle(phone_mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
               (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
    cv2.circle(phone_mask, (cx, cy), radius, 255, -1)
alpha = np.maximum(alpha, phone_mask)

# 2. local-contrast the luminance (CLAHE)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# a touch of global lift so the face sits in the sparse end of the ramp
gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)

# 3. paste onto white using the alpha mask (feathered a hair to avoid a halo)
mask = (alpha.astype(np.float32) / 255.0)
mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
out = np.clip(out, 0, 255).astype(np.uint8)

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape)
