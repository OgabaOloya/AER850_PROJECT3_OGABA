# ===============================================================
#  AER850 Project 3 - Object Masking Script
# ===============================================================

import cv2
import numpy as np
import os


# Creating an output folder 
output_folder = "outputs_step1"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------
# Loading our Image 
# ---------------------------------------------------------
image = cv2.imread("data/motherboard_image.jpeg")

# Converting the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adding Gaussian Blur to smoothen out noise before the edge detection
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Applying Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Dilation and Closing
kernel = np.ones((5, 5), np.uint8)

# Strengthening the edges by connecting thin/broken lines
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# Closing small gaps in the contour
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

# Contour detection
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering large contours
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 4000]

# Selecting the largest contour (PCB)
largest_contour = max(large_contours, key=cv2.contourArea)

# Mask creation from contour 
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Extracting the PCB using mask 
extracted = cv2.bitwise_and(image, image, mask=mask)

# ---------------------------------------------------------
# Saving the outputs for Step 1 into the folder
# ---------------------------------------------------------
cv2.imwrite(os.path.join(output_folder, "gray.jpg"), gray)
cv2.imwrite(os.path.join(output_folder, "blur.jpg"), blur)
cv2.imwrite(os.path.join(output_folder, "edges.jpg"), edges)
cv2.imwrite(os.path.join(output_folder, "edges_dilated.jpg"), edges_dilated)
cv2.imwrite(os.path.join(output_folder, "edges_closed.jpg"), edges_closed)
cv2.imwrite(os.path.join(output_folder, "mask.jpg"), mask)
cv2.imwrite(os.path.join(output_folder, "extracted.jpg"), extracted)

print("All images saved to:", output_folder)

