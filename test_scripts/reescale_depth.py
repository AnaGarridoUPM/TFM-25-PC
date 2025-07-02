# script para reescalar valores
import cv2

def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel at ({x},{y}) = {param[y, x]}")

depth = cv2.imread("0_depth_mask.png", cv2.IMREAD_UNCHANGED)

# Apply scaling for better visualization
scaled_depth = cv2.convertScaleAbs(depth, alpha=0.03)

# Save the enhanced visualization
cv2.imwrite("enhanced_depth_visualization.png", scaled_depth)

cv2.namedWindow("depth")
cv2.setMouseCallback("depth", show_pixel_value, depth)

while True:
    cv2.imshow("depth", scaled_depth)  # Display scaled depth image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
