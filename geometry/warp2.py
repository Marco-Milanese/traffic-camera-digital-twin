"""
Module for selecting points for homography computation.
Performs an "uncropped" perspective warp, ensuring the entire source image
is visible in the transformed output.
"""
import os
import cv2
import numpy as np

# --- Helper functions remain mostly unchanged ---

def calculateDestinationPoints(real_width_meters, real_height_meters, scale_ppm=20):
    """
    Returns 4 points forming a Perfect Rectangle starting at (0,0).
    These define the SCALE of the transform, but no longer define the borders.
    """
    w_pixels = int(real_width_meters * scale_ppm)
    h_pixels = int(real_height_meters * scale_ppm)
    
    return np.array([
        [0, 0],             # TL
        [w_pixels, 0],      # TR
        [w_pixels, h_pixels], # BR
        [0, h_pixels]       # BL
    ], dtype=np.float32)


def selectHomographyPoints(imagePath):
    # (Kept the same as your provided code)
    srcPoints = []
    cv2.namedWindow("Homography Point Selector")

    if not os.path.exists(imagePath):
        raise FileNotFoundError(f"Image file not found: {imagePath}")
    img = cv2.imread(imagePath)
    if img is None:
        raise ValueError(f"Failed to load image: {imagePath}")

    h, w = img.shape[:2]
    # Reduce font size slightly for better visibility on smaller images
    font_scale = 0.6
    thickness = 2

    imgPrompt1 = cv2.putText(img.copy(), "Click TOP-LEFT corner", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    # Using a mutable list to allow the closure to update image state
    current_image_container = [imgPrompt1]

    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw the confirmed point on the base image for persistence
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            
            match len(srcPoints):
                case 0:
                    srcPoints.append([x, y])
                    prompt = "Click TOP-RIGHT corner"
                case 1:
                    srcPoints.append([x, y])
                    prompt = "Click BOTTOM-RIGHT corner"
                case 2:
                    srcPoints.append([x, y])
                    prompt = "Click BOTTOM-LEFT corner"
                case 3:
                    srcPoints.append([x, y])
                    prompt = "All points selected. Press any key."
            
            # Update prompt text on a fresh copy of the image with points drawn
            shownImage = cv2.putText(img.copy(), prompt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            cv2.imshow("Homography Point Selector", shownImage)
            print(f"Point {len(srcPoints)} selected at: {x}, {y}")

    cv2.setMouseCallback("Homography Point Selector", clickEvent)
    cv2.imshow("Homography Point Selector", imgPrompt1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(srcPoints) != 4:
        raise ValueError("Did not select 4 points before exiting.")
    return srcPoints


def computeHomographyMatrix(srcPoints, dstPoints):
    srcPts = np.array(srcPoints, dtype=np.float32)
    dstPts = np.array(dstPoints, dtype=np.float32)
    # We use getPerspectiveTransform because we have exactly 4 corresponding pairs
    homographyMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    return homographyMatrix

# --- Main Execution Function (HEAVILY MODIFIED) ---

def runHomography(imagePath):
    img = cv2.imread(imagePath)
    if img is None:
        print(f"Error: Could not read image {imagePath}")
        return None

    # 1. Select points and compute initial matrix
    print("Please select 4 corners in order: TL, TR, BR, BL")
    srcPoints = selectHomographyPoints(imagePath)
    
    # Define scale: 105m x 65m field, 20 pixels per meter
    dstPoints = calculateDestinationPoints(27, 77, 20)
    H_initial = computeHomographyMatrix(srcPoints, dstPoints)

    # --- NEW LOGIC FOR UNCROPPED VIEW ---

    # 2. Get dimensions of the actual source image
    h_img, w_img = img.shape[:2]

    # 3. Define the 4 corners of the source image (not the clicked points)
    # Must be shape (-1, 1, 2) for perspectiveTransform
    img_corners_src = np.float32([
        [0, 0], 
        [w_img, 0], 
        [w_img, h_img], 
        [0, h_img]
    ]).reshape(-1, 1, 2)

    # 4. Predict where these corners will land using the initial matrix
    img_corners_dst = cv2.perspectiveTransform(img_corners_src, H_initial)

    # 5. Find the bounding box of these transformed corners
    x_coords = img_corners_dst[:, :, 0].ravel()
    y_coords = img_corners_dst[:, :, 1].ravel()

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # 6. Calculate necessary translation shift.
    # If x_min is negative, we must shift right by that amount.
    # Use floor to ensure we capture sub-pixel edge cases.
    translation_x = -np.floor(x_min) if x_min < 0 else 0
    translation_y = -np.floor(y_min) if y_min < 0 else 0

    # 7. Create the Translation Matrix
    T = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # 8. Combine matrices: Final H = Translation * Initial H
    H_final = T.dot(H_initial)

    # 9. Determine final canvas size based on the bounding box
    final_w = int(np.ceil(x_max) - np.floor(x_min))
    final_h = int(np.ceil(y_max) - np.floor(y_min))

    # 10. Perform the warp with the final matrix and new canvas size
    # Use borderMode=cv2.BORDER_TRANSPARENT if you plan to overlay, 
    # otherwise BORDER_CONSTANT (black background) is fine.
    warpedImg = cv2.warpPerspective(img, H_final, (final_w, final_h))

    # --- Saving and Display ---

    base, _ext = os.path.splitext(os.path.basename(imagePath))
    output_filename = f"{base}_full_warp.png"
    # Save in the same directory as the script for simplicity, or adjust path as needed
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    
    cv2.imwrite(output_path, warpedImg)
    print(f"Warped full image saved to: {output_path}")

    # Resize for display if the resulting image is huge
    display_img = warpedImg.copy()
    if final_w > 1280 or final_h > 800:
        scale = min(1280/final_w, 800/final_h)
        display_img = cv2.resize(display_img, None, fx=scale, fy=scale)

    cv2.imshow("Full Warped Image (Resized for display)", display_img)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return H_final

# =========================================
# UPDATE THIS PATH TO YOUR IMAGE FILE
# =========================================
image_path = r"geometry/tokyo.png" 

if os.path.exists(image_path):
    M = runHomography(image_path)
else:
    print(f"Please update 'image_path' variable. File not found: {image_path}")