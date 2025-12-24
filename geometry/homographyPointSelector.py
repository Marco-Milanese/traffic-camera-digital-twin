"""
Module for selecting points for homography computation.

Allows the user to interactively select points on an image with corresponding points in a map image.

These points can then be used to compute a homography matrix for perspective transformations.
"""
#cv2.getPerspectiveTransform(src, dst) --> src and dst are arrays of four points each, where src are the points in the image and dst are the points in the real world (e.g., corners of a square).
#cv2.findHomography(srcPoints, dstPoints) --> finds a perspective transformation between two planes.
#cv2.setMouseCallback(winname, onMouse, userdata=None) --> sets a mouse callback function for the specified window.
#def click_event(event, x, y, flags, param): --> to be passed to onMouse
#cv2.EVENT_LBUTTONDOWN	Left mouse button pressed
#cv2.EVENT_RBUTTONDOWN	Right mouse button pressed
#cv2.EVENT_MBUTTONDOWN	Middle mouse button pressed
#cv2.EVENT_LBUTTONUP	Left mouse button released
#cv2.EVENT_MOUSEMOVE	Mouse pointer moved over the window
#cv2.EVENT_LBUTTONDBLCLK	Left mouse button double-clicked
#cv2.namedWindow("Canvas")
#cv2.imshow("Canvas", img)

"""Get a source image (video stream frame) and a map image, show them side by side and allow the user 
to click corresponding points on each image. 

use right-click to remove the last selected point in case of mistakes.

click enter to finalize point selection and compute homography matrix

save matrix and start a demo in which user can select points in the source image and see where they map to 
on the map image using the computed homography matrix.
"""

import os
import cv2
import numpy as np

def findCenterPoint(points):
    """
    Given a list of points, calculate the center point (centroid).
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = sum(x_coords) / len(points)
    center_y = sum(y_coords) / len(points)
    return (center_x, center_y)

def calculateDestinationPoints(real_width_meters, real_height_meters, scale_ppm=20):
    """
    Returns 4 points forming a Perfect Rectangle starting at (0,0).
    Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    w_pixels = int(real_width_meters * scale_ppm)
    h_pixels = int(real_height_meters * scale_ppm)
    
    return np.array([
        [500, 500],             # TL
        [w_pixels, 500],      # TR
        [w_pixels, h_pixels], # BR
        [500, h_pixels]       # BL
    ], dtype=np.float32)


def selectHomographyPoints(srcImagePath, mapImagePath):

    srcPoints = []
    mapPoints = []
    cv2.namedWindow("Homography Point Selector")

    # Validate that the file exists and can be loaded
    if not os.path.exists(srcImagePath):
        raise FileNotFoundError(f"Source image file not found: {srcImagePath}")
    if not os.path.exists(mapImagePath):
        raise FileNotFoundError(f"Map image file not found: {mapImagePath}")
    

    srcImg = cv2.imread(srcImagePath)
    mapImg = cv2.imread(mapImagePath)
    if srcImg is None or mapImg is None:
        raise ValueError(f"Failed to load image: {srcImagePath} or {mapImagePath}")
    

    # Use height, width from shape safely
    srcH, srcW = srcImg.shape[:2]
    mapH, mapW = mapImg.shape[:2]
    #dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
    if srcH != mapH:
        if srcH > mapH:
            mapConcat = cv2.copyMakeBorder(mapImg, 0, srcH - mapH, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            srcConcat = srcImg
        else:
            srcConcat = cv2.copyMakeBorder(srcImg, 0, mapH - srcH, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            mapConcat = mapImg
    sideBySideImg = cv2.hconcat([srcConcat, mapConcat])

    resizeFactor = 0.6
    inverseResizeFactor = 1 / resizeFactor
    resizedSideBySideImg = cv2.resize(sideBySideImg, (int(((srcW + mapW) * resizeFactor)), (int(max(srcH, mapH) * resizeFactor)))) 
    resizedSrcW = int(srcW * resizeFactor)  # adjust width after resizing
    sourceSelecting = True  # True => next click should be on SOURCE image
    def clickEvent(event, x, y, flags, param):
        nonlocal sourceSelecting
        if event == cv2.EVENT_LBUTTONDOWN:
            # enforce alternating selection: accept clicks only on the expected side
            if sourceSelecting and x < resizedSrcW:
                srcPoints.append([x, y])
                sourceSelecting = False
            elif (not sourceSelecting) and x >= resizedSrcW:
                mapPoints.append([x - resizedSrcW, y])
                sourceSelecting = True
            # clicks on the wrong side are ignored
        elif event == cv2.EVENT_RBUTTONDOWN:
            # undo last selection: previous side is opposite of current expectation
            prev_was_src = not sourceSelecting
            if prev_was_src:
                if srcPoints:
                    srcPoints.pop()
                    sourceSelecting = True
            else:
                if mapPoints:
                    mapPoints.pop()
                    sourceSelecting = False

    done = False

    cv2.namedWindow("Homography Point Selector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Homography Point Selector", clickEvent)

    while not done:
        defaultScreen = cv2.putText(resizedSideBySideImg.copy(),
                                    ("Select a point on SOURCE image, then a point on MAP image. "
                                     "Right-click to undo last point. Press ENTER when done."),
                                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw paired points and connecting lines
        for i in range(min(len(srcPoints), len(mapPoints))):
            cv2.circle(defaultScreen, (srcPoints[i][0], srcPoints[i][1]), 5, (255, 0, 0), -1)
            cv2.circle(defaultScreen, (mapPoints[i][0] + resizedSrcW, mapPoints[i][1]), 5, (0, 255, 0), -1)
            cv2.line(defaultScreen, (srcPoints[i][0], srcPoints[i][1]),
                     (mapPoints[i][0] + resizedSrcW, mapPoints[i][1]), (0, 0, 255), 1)

        # Draw unmatched last point safely
        if len(srcPoints) > len(mapPoints):
            x, y = srcPoints[-1]
            cv2.circle(defaultScreen, (x, y), 5, (255, 0, 0), -1)
        elif len(mapPoints) > len(srcPoints):
            x, y = mapPoints[-1]
            cv2.circle(defaultScreen, (x + resizedSrcW, y), 5, (0, 255, 0), -1)

        # Show which side is expected next
        next_text = "Next: SOURCE" if sourceSelecting else "Next: MAP"
        cv2.putText(defaultScreen, next_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Homography Point Selector", defaultScreen)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            if len(srcPoints) >= 4 and len(srcPoints) == len(mapPoints):
                done = True
            else:
                print("Need at least 4 matching pairs before finishing.")
        elif key == 27:  # ESC
            print("Selection canceled.")
            srcPoints.clear(); mapPoints.clear(); done = True

    cv2.destroyAllWindows()

    # Convert to numpy arrays for downstream processing
    srcPoints = np.array(srcPoints, dtype=np.float32)
    mapPoints = np.array(mapPoints, dtype=np.float32)


    return srcPoints * inverseResizeFactor, mapPoints * inverseResizeFactor

def computeHomographyMatrix(srcPoints, dstPoints):
    
    srcPts = np.array(srcPoints, dtype=np.float32)
    dstPts = np.array(dstPoints, dtype=np.float32)
    homographyMatrix = cv2.findHomography(srcPts, dstPts)
    return homographyMatrix


def HomographyDemo(srcImgPath, mapPath, homographyMatrix, resize_factor=.7):
    srcImg = cv2.imread(srcImgPath)
    mapImg = cv2.imread(mapPath)
    if srcImg is None or mapImg is None:
        raise ValueError(f"Failed to load image: {srcImgPath} or {mapPath}")

    if homographyMatrix is None or homographyMatrix[0] is None:
        raise ValueError("Invalid homography matrix provided.")
    H = homographyMatrix[0]  # extract matrix from tuple

    inv_scale = 1.0 / resize_factor

    # Resize images to 1/4 of original (or given resize_factor)
    srcH, srcW = srcImg.shape[:2]
    mapH, mapW = mapImg.shape[:2]
    resized_src = cv2.resize(srcImg, (int(srcW * resize_factor), int(srcH * resize_factor)), interpolation=cv2.INTER_AREA)
    resized_map = cv2.resize(mapImg, (int(mapW * resize_factor), int(mapH * resize_factor)), interpolation=cv2.INTER_AREA)

    # Pad the shorter image so both heights match before concatenation
    r_src_h, r_src_w = resized_src.shape[:2]
    r_map_h, r_map_w = resized_map.shape[:2]
    if r_src_h != r_map_h:
        if r_src_h > r_map_h:
            resized_map = cv2.copyMakeBorder(resized_map, 0, r_src_h - r_map_h, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        else:
            resized_src = cv2.copyMakeBorder(resized_src, 0, r_map_h - r_src_h, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])

    side_by_side = cv2.hconcat([resized_src, resized_map])
    src_section_width = resized_src.shape[1]

    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # only respond to clicks inside the source image area
            if x < src_section_width:
                # scale clicked point back to original-size coordinates for homography
                orig_x = int(x * inv_scale)
                orig_y = int(y * inv_scale)
                srcPoint = np.array([[orig_x, orig_y]], dtype=np.float32)
                srcPointHomogeneous = cv2.perspectiveTransform(np.array([srcPoint]), H)
                mappedPoint = srcPointHomogeneous[0][0]
                mappedX_orig, mappedY_orig = mappedPoint[0], mappedPoint[1]

                # scale mapped point down to resized map coordinates for display
                mappedX_display = int(mappedX_orig * resize_factor)
                mappedY_display = int(mappedY_orig * resize_factor)

                display = side_by_side.copy()
                # mark source click (on left)
                cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
                # mark mapped point (on right, offset by src_section_width)
                cv2.circle(display, (mappedX_display + src_section_width, mappedY_display), 5, (0, 0, 255), -1)
                # draw a line from source click to mapped location
                cv2.line(display, (x, y), (mappedX_display + src_section_width, mappedY_display), (0, 255, 0), 1)

                cv2.imshow("Homography Demo", display)
                print(f"Source (display): ({x}, {y}) -> Source (orig): ({orig_x}, {orig_y}) mapped to Map (orig): ({mappedX_orig:.2f}, {mappedY_orig:.2f}) -> Map (display): ({mappedX_display}, {mappedY_display})")

    cv2.namedWindow("Homography Demo", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Homography Demo", clickEvent)
    cv2.imshow("Homography Demo", side_by_side)
    print("Click on the source image (left) to see mapped points on the map (right). Press ESC to exit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    srcPoints, mapPoints = selectHomographyPoints(r"geometry/ShibuyaScramble.png", r"geometry/ShibuyaScrambleMap.png")
    H = computeHomographyMatrix(srcPoints, mapPoints)
    np.save(r"geometry/ScrambleHomography.npy", H[0])
    np.save(r"geometry/ScrambleMask.npy", H[1])
    H = [np.load(r"geometry/ScrambleHomography.npy"), np.load(r"geometry/ScrambleMask.npy")]
    HomographyDemo(r"geometry/ShibuyaScramble.png", r"geometry/ShibuyaScrambleMap.png", H)