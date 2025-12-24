import os
import cv2
from ultralytics import YOLO
import yt_dlp
import numpy as np

# Configuration
MODEL_PATH = "yolo11n-visdrone.pt"  # visdrone-trained weights (change if needed)
# Classes to visualize on the map (lowercase)
MONITORED_CLASSES = {"person", "people", "pedestrian", "car", "truck", "bus", "van", "motorbike", "motorcycle", "bicycle", "bike"}
# Colors for classes (BGR)
CLASS_COLORS = {
    "person": (0, 255, 0),      # green
    "car": (0, 0, 255),         # red
    "truck": (255, 0, 0),       # blue
    "bus": (0, 165, 255),       # orange
    "van": (255, 255, 0),       # cyan-like
    "motorbike": (255, 0, 255), # magenta
    "motorcycle": (255, 0, 255),
    "bicycle": (200, 100, 255), # light purple
    "bike": (200, 100, 255)
}
MAP_MARKER_RADIUS = 8
EDGE_MARKER_RADIUS = 5
DEFAULT_CLASS_COLOR = (0, 255, 255)  # yellow for unknowns


def get_video_stream_url(youtube_url):
    """
    Extracts the direct raw stream URL using yt-dlp.
    """
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Select best quality
        'quiet': True,
        'noplaylist': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def run_yolo_on_stream(youtube_url):
    # 1. Load the YOLO model
    # The model will download automatically on first run
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    # Print class names for debugging
    try:
        print(f"Model classes (sample): {list(model.names.items())[:20]}")
    except Exception:
        pass


    # Try to load homography and map for Tokyo visualization
    map_img = None
    H = None
    src_reference = None
    homography_path = "geometry/tokyoHomography.npy"
    map_path = "geometry/tokyoMap.png"
    src_reference_path = "geometry/tokyo.png"
    if os.path.exists(homography_path):
        try:
            H = np.load(homography_path, allow_pickle=True)
            if H is None or getattr(H, "size", 0) == 0:
                H = None
        except Exception as e:
            print(f"Warning: Could not load homography from {homography_path}: {e}")
            H = None
    else:
        H = None

    # Normalize H to be a 3x3 float32 matrix if possible
    if H is not None:
        print(f"Loaded homography: type={type(H)}, shape={getattr(H, 'shape', None)}, dtype={getattr(H, 'dtype', None)}")
        # Handle cases where H might be saved as a tuple (homography, mask) or wrapped
        try:
            # If H is an array of objects or has extra nesting, try to extract the first ndarray
            if isinstance(H, np.ndarray) and H.dtype == object and H.size > 0 and isinstance(H.flat[0], np.ndarray):
                H = H.flat[0]
            # If H is a 1x3x3 array (e.g., saved with extra axis), squeeze it
            if isinstance(H, np.ndarray) and H.ndim == 3 and H.shape[0] == 1:
                H = H[0]
            # If it's a tuple or list, pick the first element that is ndarray
            if isinstance(H, (tuple, list)):
                for el in H:
                    if isinstance(el, np.ndarray) and el.shape == (3, 3):
                        H = el
                        break
            if not (isinstance(H, np.ndarray) and H.shape == (3, 3)):
                print(f"Warning: Homography has unexpected shape {getattr(H, 'shape', None)}; map visualization disabled.")
                H = None
            else:
                H = H.astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not normalize homography: {e}")
            H = None

    if H is not None and os.path.exists(map_path):
        map_img = cv2.imread(map_path)
        if map_img is None:
            print(f"Warning: Failed to load map image at {map_path}")
            map_img = None
        if os.path.exists(src_reference_path):
            src_reference = cv2.imread(src_reference_path)
            if src_reference is None:
                src_reference = None
    else:
        if H is None:
            print("Warning: Homography not found; map visualization disabled.")
        else:
            print("Warning: Map image not found; map visualization disabled.")

    if map_img is not None:
        print(f"Loaded map image shape: {map_img.shape}")
        cv2.namedWindow("Map Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Map Visualization", map_img.shape[1], map_img.shape[0])

    # 2. Get the direct stream URL
    print(f"Extracting stream URL from {youtube_url}...")
    try:
        stream_url = get_video_stream_url(youtube_url)
    except Exception as e:
        print(f"Error extracting stream: {e}")
        return

    # 3. Open the Video Capture
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Starting inference. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Stream ended or failed to read frame.")
            break

        # 4. Run YOLOv11 Inference
        # stream=True is efficient for video loops
        results = list(model(frame, stream=True))
        print(f"Debug: got {len(results)} result(s) for frame")

        # 5. Visualize Results
        for i, result in enumerate(results):
            # Plot the detections on the frame
            annotated_frame = result.plot()
            print(f"Debug: result[{i}] boxes={len(getattr(result, 'boxes', []))}")

            # Display the frame
            cv2.imshow("YOLO11n Live Stream Detection", annotated_frame)

        # If map visualization is available, map people and cars to the map image
        if map_img is not None and H is not None:
            print("Debug: mapping detections to map")
            map_vis = map_img.copy()
            # Count detected classes this frame for debugging (lowercased names)
            detection_counts = {}
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                try:
                    cls_ids_tmp = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    cls_ids_tmp = np.array([int(c) for c in boxes.cls])
                for cid in cls_ids_tmp:
                    raw_name = model.names.get(int(cid), str(int(cid)))
                    if isinstance(raw_name, bytes):
                        try:
                            raw_name = raw_name.decode('utf-8')
                        except Exception:
                            raw_name = str(raw_name)
                    name = str(raw_name).lower()
                    detection_counts[name] = detection_counts.get(name, 0) + 1
            print(f"Detections this frame (summary): {detection_counts}")

            # Reference size (image used when homography was computed). If not available, assume current frame size.
            if src_reference is not None:
                ref_h, ref_w = src_reference.shape[:2]
            else:
                ref_h, ref_w = frame.shape[:2]

            frame_h, frame_w = frame.shape[:2]

            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                # Convert boxes to numpy arrays (supports both torch tensors and lists)
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    xyxy = np.array([b.tolist() for b in boxes.xyxy])
                    cls_ids = np.array([int(c) for c in boxes.cls])

                for bbox, cls_id in zip(xyxy, cls_ids):
                    x1, y1, x2, y2 = bbox[:4]
                    # ensure class name is lowercase for comparison
                    class_name = model.names.get(int(cls_id), str(int(cls_id))).lower()
                    if class_name not in MONITORED_CLASSES:
                        continue

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # Scale center point to reference image used for homography
                    scale_x = ref_w / float(frame_w)
                    scale_y = ref_h / float(frame_h)
                    src_x = center_x * scale_x
                    src_y = center_y * scale_y


                    src_pt = np.array([[[src_x, src_y]]], dtype=np.float32)
                    try:
                        mapped = cv2.perspectiveTransform(src_pt, H)
                        mx = int(round(mapped[0][0][0]))
                        my = int(round(mapped[0][0][1]))
                        in_bounds = (0 <= mx < map_vis.shape[1] and 0 <= my < map_vis.shape[0])
                        print(f"Mapped {class_name}: src=({src_x:.1f},{src_y:.1f}) -> mapped=({mx},{my}) in_bounds={in_bounds}")
                        color = CLASS_COLORS.get(class_name, DEFAULT_CLASS_COLOR)
                        if in_bounds:
                            cv2.circle(map_vis, (mx, my), MAP_MARKER_RADIUS, color, -1)
                        else:
                            # Optionally clamp to map edges for visibility
                            cx = max(0, min(mx, map_vis.shape[1]-1))
                            cy = max(0, min(my, map_vis.shape[0]-1))
                            cv2.circle(map_vis, (cx, cy), EDGE_MARKER_RADIUS, DEFAULT_CLASS_COLOR, -1)
                    except Exception as e:
                        print(f"Error transforming point {src_pt} through H: {e}")
                        # don't crash on transform errors; continue to next detection

            cv2.imshow("Map Visualization", map_vis)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

 

if __name__ == "__main__":
    # Replace with a valid YouTube Live Stream URL
    # Example: A 24/7 nature or traffic stream works best
    target_url = "https://www.youtube.com/watch?v=6dp-bvQ7RWo" 
    
    run_yolo_on_stream(target_url)