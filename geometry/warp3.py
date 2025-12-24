"""
Module for selecting points for homography computation.

Allows the user to interactively select 4 points on an image corresponding to a perfect square in the real world.

These points can then be used to compute a homography matrix for perspective transformations.
"""
#cv2.getPerspectiveTransform(src, dst) --> src and dst are arrays of four points each, where src are the points in the image and dst are the points in the real world (e.g., corners of a square).
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


def selectHomographyPoints(imagePath):

    srcPoints = []
    cv2.namedWindow("Homography Point Selector")

    # Validate that the file exists and can be loaded
    if not os.path.exists(imagePath):
        raise FileNotFoundError(f"Image file not found: {imagePath}")
    img = cv2.imread(imagePath)
    if img is None:
        raise ValueError(f"Failed to load image: {imagePath}")

    # Use height, width from shape safely
    h, w = img.shape[:2]

    imgPrompt1 = cv2.putText(img.copy(), "Click TOP-LEFT corner", (int(w/2) - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            match len(srcPoints):
                case 0:
                    srcPoints.append([x, y])
                    shownImage = cv2.putText(img.copy(), "Click TOP-RIGHT corner", (int(w/2) - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    shownImage = cv2.circle(shownImage, (x, y), 5, (0, 0, 255), -1)
                    print("Point 1 selected at: ", x, y)
                    cv2.imshow("Homography Point Selector", shownImage)
                case 1:
                    srcPoints.append([x, y])
                    shownImage = cv2.putText(img.copy(), "Click BOTTOM-RIGHT corner", (int(w/2) - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    shownImage = cv2.circle(shownImage, srcPoints[0], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, (x, y), 5, (0, 0, 255), -1)
                    print("Point 2 selected at: ", x, y)
                    cv2.imshow("Homography Point Selector", shownImage)
                case 2:
                    srcPoints.append([x, y])
                    shownImage = cv2.putText(img.copy(), "Click BOTTOM-LEFT corner", (int(w/2) - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    shownImage = cv2.circle(shownImage, srcPoints[0], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, srcPoints[1], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, (x, y), 5, (0, 0, 255), -1)
                    print("Point 3 selected at: ", x, y)
                    cv2.imshow("Homography Point Selector", shownImage)
                case 3:
                    srcPoints.append([x, y])
                    shownImage = cv2.putText(img.copy(), "All 4 points selected", (int(w/2) - 40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    shownImage = cv2.circle(shownImage, srcPoints[0], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, srcPoints[1], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, srcPoints[2], 5, (0, 0, 255), -1)
                    shownImage = cv2.circle(shownImage, (x, y), 5, (0, 0, 255), -1)
                    print("Point 4 selected at: ", x, y)
                    cv2.imshow("Homography Point Selector", shownImage)
                    print("All 4 points selected: ", srcPoints)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

    cv2.setMouseCallback("Homography Point Selector", clickEvent)
    cv2.imshow("Homography Point Selector", imgPrompt1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return srcPoints

def computeHomographyMatrix(srcPoints, dstPoints):
    """
    Given source and destination points, compute the homography matrix.
    """
    srcPts = np.array(srcPoints, dtype=np.float32)
    dstPts = np.array(dstPoints, dtype=np.float32)
    homographyMatrix = cv2.getPerspectiveTransform(srcPts, dstPts)
    return homographyMatrix
# Example call -- change path to your image file or pass a full path

def runHomography(imagePath):
    srcPoints = selectHomographyPoints(imagePath)
    dstPoints = calculateDestinationPoints(28.75, 15.24, 100)
    homographyMatrix = computeHomographyMatrix(srcPoints, dstPoints)
    img = cv2.imread(imagePath)
    # Use destination rectangle size (width from TR.x, height from BR.y)
    dst_w = int(dstPoints[1][0])
    dst_h = int(dstPoints[2][1])
    warpedImg = cv2.warpPerspective(img, homographyMatrix, (7000, 4000))

    # Save warped image beside input with suffix _warped.png
    base, _ext = os.path.splitext(os.path.basename(imagePath))
    output_filename = f"{base}_warped.png"
    output_path = os.path.join(os.path.dirname(imagePath), output_filename)
    cv2.imwrite(output_path, warpedImg)
    print(f"Warped image saved to: {output_path}")

    cv2.imshow("Warped Image", warpedImg)
    cv2.waitKey(0)
    return homographyMatrix


def auto_compute_homography_from_images(src_img_path, map_img_path, min_matches=8, debug=False):
    """Automatically compute homography from source image to map image using ORB feature matching.

    Returns a homography matrix (3x3) or raises ValueError if insufficient matches are found.
    """
    if not os.path.exists(src_img_path) or not os.path.exists(map_img_path):
        raise FileNotFoundError("Source image or map image not found.")

    src_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
    map_img = cv2.imread(map_img_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(src_img, None)
    kp2, des2 = orb.detectAndCompute(map_img, None)

    if des1 is None or des2 is None:
        raise ValueError("Failed to compute descriptors for one of the images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if debug:
        print(f"Found {len(good)} good matches")

    if len(good) < min_matches:
        raise ValueError(f"Not enough good matches found: {len(good)} (required {min_matches})")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography estimation failed.")

    if debug:
        inliers = mask.sum() if mask is not None else 0
        print(f"Homography estimated with {inliers} inliers")

    return H


def create_side_by_side(src_img, map_img):
    """Return a combined image with the source on the left and map on the right."""
    h1, w1 = src_img.shape[:2]
    h2, w2 = map_img.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = src_img
    canvas[:h2, w1:w1 + w2] = map_img
    return canvas, w1, w2


def compute_homography_from_points(src_pts_orig, dst_pts_orig, debug=False):
    """Compute homography from lists of corresponding original-image points.

    src_pts_orig and dst_pts_orig should be iterables of (x,y) in the ORIGINAL image coordinate spaces.
    Returns a (3x3) homography matrix or raises ValueError if computation fails.
    """
    if len(src_pts_orig) < 4 or len(dst_pts_orig) < 4:
        raise ValueError("At least 4 correspondences are required to compute a reliable homography.")
    src_arr = np.float32(src_pts_orig).reshape(-1,1,2)
    dst_arr = np.float32(dst_pts_orig).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography computation failed")
    if debug:
        inliers = mask.sum() if mask is not None else 0
        print(f"Computed homography with {inliers} inliers out of {len(src_pts_orig)} correspondences")
    return H


def interactive_projection_test(src_img_path, map_img_path, homography=None, display_scale=1.2, manual_correspondences=False):
    """Show source and map side-by-side; clicking on source projects the point onto the map using the homography.

    If manual_correspondences is True the UI will collect matching point pairs (source then map) and compute a homography
    when you press 'c'. Use right-click to remove the last pair. Once computed the interface switches to projection mode.

    Left click: depends on mode
      - manual collection mode: click source (left), then click corresponding map point (right), repeat
      - projection mode: click source to project to map
    Keys:
      - 'c' compute homography from collected pairs (manual mode only)
      - 'u' undo last pair (manual mode only)
      - Right-click: clear all points (projection mode)
      - 'q' or ESC: quit
    """
    if not os.path.exists(src_img_path) or not os.path.exists(map_img_path):
        raise FileNotFoundError("Source image or map image not found for test.")

    src_bgr = cv2.imread(src_img_path)
    map_bgr = cv2.imread(map_img_path)

    # Resize map to a sensible size if very large (but keep aspect ratio)
    max_dim = 1600
    def _resize_if_needed(img):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        return img

    src_bgr = _resize_if_needed(src_bgr)
    map_bgr = _resize_if_needed(map_bgr)

    # Apply display scaling (can be >1.0 to make the test visuals larger)
    if display_scale != 1.0:
        src_display = cv2.resize(src_bgr, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
        map_display = cv2.resize(map_bgr, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_LINEAR)
    else:
        src_display = src_bgr
        map_display = map_bgr

    # create combined canvas from the display-sized images
    canvas, x_offset, _ = create_side_by_side(src_display, map_display)
    canvas_display = canvas.copy()
    src_w_display = x_offset

    # scaling factors to convert between display coordinates and original image coordinates
    src_scale = src_display.shape[1] / src_bgr.shape[1]
    map_scale = map_display.shape[1] / map_bgr.shape[1]

    # state for manual correspondence collection
    collecting = manual_correspondences
    expect = 'src'  # when collecting: expect 'src' then 'map'
    collected_src_orig = []
    collected_map_orig = []
    collected_pairs_display = []  # list of ((sx,sy),(mx_disp,my_disp)) for drawing

    # state for projection mode (points to draw)
    proj_src_display = []
    proj_map_display = []

    def draw(mode_text=None):
        nonlocal canvas_display
        canvas_display = canvas.copy()
        # header instructions
        header = mode_text if mode_text is not None else ("Manual mode: click source then map; 'c' compute; 'u' undo; 'q' quit" if collecting else "Projection mode: click source to project; right-click to clear; 'q' to quit")
        cv2.putText(canvas_display, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Draw collected correspondences
        for idx, ((sx,sy),(mx_disp,my_disp)) in enumerate(collected_pairs_display):
            cv2.circle(canvas_display, (int(sx), int(sy)), 6, (0,255,0), -1)
            cv2.circle(canvas_display, (int(mx_disp), int(my_disp)), 6, (255,0,255), -1)
            # line between the two
            cv2.line(canvas_display, (int(sx), int(sy)), (int(mx_disp), int(my_disp)), (200,200,0), 1)
            cv2.putText(canvas_display, str(idx+1), (int(sx)+8, int(sy)+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw projected points (blue/red)
        for p in proj_src_display:
            cv2.circle(canvas_display, (int(p[0]), int(p[1])), 6, (0,0,255), -1)
        for p in proj_map_display:
            cv2.circle(canvas_display, (int(p[0]), int(p[1])), 6, (255,0,0), -1)

    def on_mouse(event, x, y, flags, param):
        nonlocal collecting, expect, collected_src_orig, collected_map_orig, collected_pairs_display
        nonlocal proj_src_display, proj_map_display, homography
        if collecting:
            if event == cv2.EVENT_LBUTTONDOWN:
                # if expecting src, accept only left side
                if expect == 'src' and x < src_w_display:
                    # convert to original coords
                    orig_x = x / src_scale
                    orig_y = y / src_scale
                    # record display src location
                    src_disp = (x, y)
                    # temporarily store and ask for map click
                    expect = 'map'
                    collected_src_temp = (orig_x, orig_y, src_disp)
                    # store temp on the function object for map click
                    on_mouse.collected_src_temp = collected_src_temp
                    print(f"Collected source display {src_disp} -> orig ({orig_x:.2f},{orig_y:.2f}). Now click matching point on the map.")
                    draw(mode_text="Now click matching point on the MAP side (right)")
                    cv2.imshow("Projection Test", canvas_display)
                elif expect == 'map' and x >= src_w_display:
                    # map click
                    # convert to original map coords
                    mx_disp = x - src_w_display
                    my_disp = y
                    mx_orig = (mx_disp) / map_scale
                    my_orig = my_disp / map_scale
                    # retrieve temp src
                    temp = getattr(on_mouse, 'collected_src_temp', None)
                    if temp is None:
                        print("No source point recorded - click a source point first.")
                        return
                    orig_sx, orig_sy, src_disp = temp
                    collected_src_orig.append((orig_sx, orig_sy))
                    collected_map_orig.append((mx_orig, my_orig))
                    # store display pair for drawing (map display coords include offset)
                    collected_pairs_display.append((src_disp, (mx_disp + src_w_display, my_disp)))
                    print(f"Collected map display ({mx_disp + src_w_display},{my_disp}) -> orig ({mx_orig:.2f},{my_orig:.2f}). Pair saved.")
                    # reset expectation
                    expect = 'src'
                    # remove temp
                    on_mouse.collected_src_temp = None
                    draw()
                    cv2.imshow("Projection Test", canvas_display)
                else:
                    # clicked on wrong side
                    print("Click on the expected side: source (left) or map (right) depending on the prompt.")
            elif event == cv2.EVENT_RBUTTONDOWN:
                # right-click clears collected pairs
                collected_src_orig = []
                collected_map_orig = []
                collected_pairs_display = []
                expect = 'src'
                print("Cleared collected correspondences")
                draw()
                cv2.imshow("Projection Test", canvas_display)
        else:
            # projection mode behavior
            if event == cv2.EVENT_LBUTTONDOWN:
                if x < src_w_display:
                    proj_src_display.append((x,y))
                    # convert to original src coords, project, then convert map to display
                    orig_x = x / src_scale
                    orig_y = y / src_scale
                    src_pt_orig = np.array([[[orig_x, orig_y]]], dtype=np.float32)
                    map_pt_orig = cv2.perspectiveTransform(src_pt_orig, homography)
                    mx_orig, my_orig = map_pt_orig[0][0]
                    mx_disp = mx_orig * map_scale + src_w_display
                    my_disp = my_orig * map_scale
                    proj_map_display.append((mx_disp, my_disp))
                    print(f"Projected src ({orig_x:.2f},{orig_y:.2f}) -> map orig ({mx_orig:.2f},{my_orig:.2f})")
                    draw()
                    cv2.imshow("Projection Test", canvas_display)
            elif event == cv2.EVENT_RBUTTONDOWN:
                proj_src_display = []
                proj_map_display = []
                print("Cleared projected points")
                draw()
                cv2.imshow("Projection Test", canvas_display)

    draw()
    cv2.namedWindow("Projection Test", cv2.WINDOW_NORMAL)
    try:
        cw, ch = canvas.shape[1], canvas.shape[0]
        cv2.resizeWindow("Projection Test", int(cw * 1.05), int(ch * 1.05))
    except Exception:
        pass

    cv2.setMouseCallback("Projection Test", on_mouse)
    cv2.imshow("Projection Test", canvas_display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        if collecting:
            if key == ord('u'):
                # undo last pair
                if collected_src_orig and collected_map_orig:
                    collected_src_orig.pop()
                    collected_map_orig.pop()
                    collected_pairs_display.pop()
                    expect = 'src'
                    print("Undid last pair")
                    draw()
                    cv2.imshow("Projection Test", canvas_display)
            elif key == ord('c'):
                # compute homography from collected pairs
                try:
                    if len(collected_src_orig) < 4:
                        print(f"Need at least 4 correspondences to compute homography (have {len(collected_src_orig)})")
                        continue
                    H_new = compute_homography_from_points(collected_src_orig, collected_map_orig, debug=True)
                    homography = H_new
                    collecting = False
                    print("Computed homography from manual correspondences; switching to projection mode.")
                    draw()
                    cv2.imshow("Projection Test", canvas_display)
                except Exception as e:
                    print("Failed to compute homography:", e)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Demo/test: attempt to run interactive test. If only source image exists, synthesize a simple top-down map.
    src_demo = os.path.join(os.path.dirname(__file__), "tokyo.png")
    map_demo = os.path.join(os.path.dirname(__file__), "tokyoMap.png")

    if not os.path.exists(src_demo):
        print("No demo source image found. Place 'tokyo.png' in the geometry folder to run the demo.")
    else:
        if not os.path.exists(map_demo):
            # create a synthetic top-down map by warping the source into a rectangle
            print("'tokyo_map.png' not found, creating a synthetic map from 'tokyo.png'...")
            src_img = cv2.imread(src_demo)
            h, w = src_img.shape[:2]
            # source corners
            src_corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            # destination corners create a slightly scaled rectangle (top-down like)
            dst_w, dst_h = int(w*0.8), int(h*0.6)
            dst_corners = np.float32([[50,50],[dst_w-50,50],[dst_w-50,dst_h-50],[50,dst_h-50]])
            H_syn = cv2.getPerspectiveTransform(src_corners, dst_corners)
            synthetic_map = cv2.warpPerspective(src_img, H_syn, (dst_w, dst_h))
            cv2.imwrite(map_demo, synthetic_map)
            print(f"Synthetic map written to {map_demo}")

        try:
            # Ask the user whether to manually select correspondences first
            ans = input("Run manual correspondence selection first? [y/N]: ").strip().lower()
            if ans in ('y', 'yes'):
                interactive_projection_test(src_demo, map_demo, homography=None, manual_correspondences=True)
            else:
                H = auto_compute_homography_from_images(src_demo, map_demo, debug=True)
                print("Homography (estimated):\n", H)
                interactive_projection_test(src_demo, map_demo, homography=H)
        except Exception as e:
            print("Demo failed:", e)
