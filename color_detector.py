"""
Real-Time Color Detection System - Phase 3 prototype
CSSE 4210 - Cassius Williams

Captures live RGB video from an external USB camera, converts each frame to HSV,
thresholds for a target color, cleans the mask, and draws bounding boxes around
detected regions. Detection events and session records are stored in SQLite.

Keys:
  1-4 : switch target color (red / green / blue / yellow)
  t   : toggle HSV tuning trackbars
  s   : save a snapshot (raw frame + mask) to ./captures
  q   : quit
"""

# argparse allows the user to provide settings when starting the program.
# os is used to create folders for saved images.
# time is used to calculate the program's FPS.
# datetime is used to create timestamps for saved snapshots.
import argparse
import os
import time
from datetime import datetime

# OpenCV handles camera input, image processing, and displaying windows.
import cv2

# NumPy is used to create arrays for HSV color ranges and image processing.
import numpy as np

# Imports the database class used to save detection events and session information.
from storage import DetectionStore


# HSV color ranges determine which pixels belong to each target color.
# HSV stands for Hue, Saturation, and Value.
# Hue represents the color, Saturation represents its intensity,
# and Value represents its brightness.
#
# OpenCV uses these HSV ranges:
# Hue: 0-179
# Saturation: 0-255
# Value: 0-255
#
# Each color has a minimum and maximum HSV value.
# Red has two ranges because red appears at both ends of the HSV hue scale.
COLOR_RANGES = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "green":  [((36, 80, 60), (85, 255, 255))],
    "blue":   [((94, 80, 60), (126, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}


# These colors determine what color the bounding boxes will be.
# OpenCV uses BGR instead of RGB when defining colors.
# For example, (0, 0, 255) represents red.
BOX_COLORS = {"red": (0, 0, 255), "green": (0, 255, 0),
              "blue": (255, 0, 0), "yellow": (0, 255, 255)}


# Connects keyboard number keys to the colors they select.
# ord() converts a character into its keyboard character code.
# Pressing 1 selects red, 2 selects green, and so on.
KEY_TO_COLOR = {ord("1"): "red", ord("2"): "green",
                ord("3"): "blue", ord("4"): "yellow"}

# This is the name of the window that displays the HSV adjustment trackbars.
TUNE_WIN = "HSV Tuning"

# Creates trackbars that allow the user to manually adjust HSV values.


def make_trackbars(color):

    # Get the starting minimum and maximum HSV values for the selected color.
    # These values are used as the initial positions of the trackbars.
    lo, hi = COLOR_RANGES[color][0]

    # Create a separate window for the HSV trackbars.
    cv2.namedWindow(TUNE_WIN)

    # Create six trackbars:
    # H min / H max = minimum and maximum Hue.
    # S min / S max = minimum and maximum Saturation.
    # V min / V max = minimum and maximum Value (brightness).
    #
    # The last number determines the maximum position of each trackbar.
    # The lambda function is a placeholder callback that does nothing.
    for name, val, mx in [("H min", lo[0], 179), ("S min", lo[1], 255), ("V min", lo[2], 255),
                          ("H max", hi[0], 179), ("S max", hi[1], 255), ("V max", hi[2], 255)]:
        cv2.createTrackbar(name, TUNE_WIN, val, mx, lambda _: None)

# Reads the current positions of the HSV trackbars.


def read_trackbars():

    # This small function makes it easier to read a trackbar by its name.
    # getTrackbarPos() returns the current value selected by the user.
    def g(n): return cv2.getTrackbarPos(n, TUNE_WIN)

    # Return the minimum and maximum HSV values as a list.
    # These values will be used to create the color detection mask.
    return [((g("H min"), g("S min"), g("V min")), (g("H max"), g("S max"), g("V max")))]

# Creates a mask that highlights pixels matching the selected HSV color ranges.


def build_mask(hsv, ranges):

    # Start with an empty mask.
    # The mask will be filled with the results of the color detection.
    mask = None

    # Loop through every HSV range provided for the selected color.
    # This is important for red because it requires two separate ranges.
    for lo, hi in ranges:

        # inRange() checks every pixel in the HSV image.
        # Pixels inside the selected HSV range become white.
        # Pixels outside the range become black.
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        # If this is the first range, use its mask as the starting mask.
        # Otherwise, combine the new mask with the existing one.
        # This allows multiple HSV ranges to be used together.
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # Create a 5x5 kernel.
    # A kernel is a small matrix used to clean up the mask.
    kernel = np.ones((5, 5), np.uint8)

    # Opening removes small white spots that may be caused by noise.
    # This helps remove unwanted detections.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing fills small black holes inside white detected areas.
    # This helps make detected objects look more complete.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Return the cleaned mask.
    return mask

# Finds objects in the mask and ignores objects that are too small.


def find_regions(mask, min_area):

    # findContours() identifies the outlines of white objects in the mask.
    # RETR_EXTERNAL only retrieves the outer outlines of detected objects.
    # CHAIN_APPROX_SIMPLE reduces the number of points stored for each outline.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Only return objects whose contour area is greater than or equal
    # to the minimum area specified by the user.
    # This prevents tiny spots and noise from being treated as objects.
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def main():

    # Create an argument parser.
    # This allows settings to be changed when running the program
    # instead of having to edit the source code.
    ap = argparse.ArgumentParser()

    # Choose which camera the program should use.
    # Camera 0 is usually the built-in camera.
    # Camera 1 is usually an external USB camera.
    ap.add_argument("--camera", type=int, default=1,
                    help="camera index (0 = built-in, 1 = usually external USB)")

    # Set the requested camera resolution.
    # The default resolution is 640 pixels wide and 480 pixels tall.
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    # Select which color the program should detect when it starts.
    # Red is the default color.
    # choices prevents the user from entering an unsupported color.
    ap.add_argument("--color", default="red", choices=COLOR_RANGES.keys())

    # Set the minimum contour area required to detect an object.
    # Objects smaller than 800 pixels will be ignored by default.
    ap.add_argument("--min-area", type=int, default=800,
                    help="minimum contour area in pixels")

    # Choose the SQLite database file used to store detection information.
    ap.add_argument("--db", default="detections.db")

    # Read the arguments provided when the program starts.
    args = ap.parse_args()

    # Open the camera using the selected camera index.
    cap = cv2.VideoCapture(args.camera)

    # Check whether the camera successfully opened.
    # If it did not, stop the program and display an error message.
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 0.")

    # Set the requested camera width, height, and frame rate.
    # The camera may not support every requested setting.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get the actual resolution being used by the camera.
    # int() converts the returned values into whole numbers.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Display the camera information in the terminal.
    # This includes the camera index, resolution, and reported FPS.
    print(
        f"[camera] index={args.camera} resolution={w}x{h} reported_fps={cap.get(cv2.CAP_PROP_FPS):.1f}")

    # Create a connection to the SQLite database.
    # This allows the program to save detection and session information.
    store = DetectionStore(args.db)

    # Start a new database session.
    # The session stores information about this particular program run.
    # The returned session_id identifies the current session.
    session_id = store.start_session(args.camera, w, h)

    # Create a folder called captures if it does not already exist.
    # Saved camera frames and masks will be placed inside this folder.
    os.makedirs("captures", exist_ok=True)

    # Set the starting color using the command-line argument.
    # The tuning variable determines whether HSV trackbars are enabled.
    color, tuning = args.color, False

    # was_detected remembers whether an object was detected in the previous frame.
    # This helps prevent the same object from being counted every single frame.
    was_detected = False

    # Keep track of how many separate detection events occur.
    detection_count = 0

    # These variables are used to calculate the frames per second (FPS).
    # frames counts processed frames.
    # fps stores the current calculated frame rate.
    # t_prev stores the time of the previous FPS calculation.
    frames, fps, t_prev = 0, 0.0, time.time()

    # These variables help calculate the average FPS for the entire session.
    # fps_sum stores the total of all FPS measurements.
    # fps_n counts how many FPS measurements have been taken.
    fps_sum, fps_n = 0.0, 0

    # Main program loop.
    # Everything inside this loop runs repeatedly while the camera is active.
    while True:

        # Read the next frame from the camera.
        # ok tells us whether the frame was successfully captured.
        # frame contains the actual image.
        ok, frame = cap.read()

        # If the camera fails to provide a frame, display an error and exit.
        if not ok:
            print("[camera] frame grab failed")
            break

        # PRE-PROCESSING
        # Apply Gaussian Blur to reduce small amounts of image noise.
        # The (7, 7) value controls the size of the blur filter.
        # A larger filter generally creates more blur.
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)

        # Convert the blurred image from BGR to HSV.
        # HSV makes it easier to identify specific colors.
        # BGR is the default color format used by OpenCV.
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # COLOR DETECTION
        # If HSV tuning is enabled, use the values from the trackbars.
        # Otherwise, use the default HSV ranges for the selected color.
        ranges = read_trackbars() if tuning else COLOR_RANGES[color]

        # Create the black-and-white mask using the selected HSV ranges.
        # Then find the objects that appear in the mask.
        mask = build_mask(hsv, ranges)
        regions = find_regions(mask, args.min_area)

        # Create a copy of the original camera frame.
        # Drawing on this copy keeps the original frame unchanged.
        output = frame.copy()

        # Loop through all detected objects.
        # Each object is represented by a contour.
        for c in regions:

            # boundingRect() finds a rectangle that surrounds the object.
            # x and y represent the top-left corner.
            # bw and bh represent the rectangle's width and height.
            x, y, bw, bh = cv2.boundingRect(c)

            # Draw a rectangle around the detected object.
            # BOX_COLORS[color] selects the correct rectangle color.
            # The final value (2) controls the rectangle thickness.
            cv2.rectangle(output, (x, y), (x + bw, y + bh),
                          BOX_COLORS[color], 2)

            # Display the name of the detected color above the rectangle.
            # FONT_HERSHEY_SIMPLEX determines the text style.
            # The 0.6 value controls the text size.
            cv2.putText(output, color, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLORS[color], 2)

        # Check whether at least one object was found.
        # If regions contains an object, detected becomes True.
        # Otherwise, detected becomes False.
        detected = len(regions) > 0

        # DETECTION LOGGING
        # Only record a detection when the program changes from
        # not detecting an object to detecting an object.
        # This prevents the same object from being counted repeatedly
        # while it remains visible in the camera.
        if detected and not was_detected:

            # Increase the total number of detection events.
            detection_count += 1

            # Find the largest object among all detected regions.
            # contourArea() calculates the area of each detected object.
            largest = max(regions, key=cv2.contourArea)

            # Get the bounding rectangle of the largest object.
            x, y, bw, bh = cv2.boundingRect(largest)

            # Save the detection information to the SQLite database.
            # This includes:
            # - Current session ID
            # - Detected color
            # - Number of detected objects
            # - Area of the largest object
            # - Bounding box coordinates
            # - Current FPS
            store.log_detection(session_id, color, len(regions), int(cv2.contourArea(largest)),
                                (x, y, bw, bh), fps)

        # Remember the detection result from the current frame.
        # This value will be checked during the next loop iteration.
        was_detected = detected

        # FPS CALCULATION
        # Increase the number of processed frames.
        frames += 1

        # Get the current time.
        now = time.time()

        # Update the FPS calculation approximately once every second.
        if now - t_prev >= 1.0:

            # Calculate FPS by dividing the number of processed frames
            # by the amount of time that has passed.
            fps = frames / (now - t_prev)

            # Add this FPS measurement to the running total.
            # Increase the number of measurements taken.
            fps_sum, fps_n = fps_sum + fps, fps_n + 1

            # Reset the frame counter and update the previous timestamp.
            frames, t_prev = 0, now

        # DISPLAY INFORMATION
        # Choose the status message based on whether an object was detected.
        if detected:
            status = f"{color.upper()} FOUND"
        else:
            status = f"Searching for {color}..."

        # Display the detection status in the top-left corner.
        # Green text means an object was found.
        # Gray text means the program is still searching.
        cv2.putText(output, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if detected else (200, 200, 200), 2)

        # Display the current FPS and total number of detection events.
        # h - 15 positions the text near the bottom of the camera window.
        cv2.putText(output, f"FPS: {fps:.1f}  Detections: {detection_count}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the processed camera frame with bounding boxes.
        cv2.imshow("Color Detection", output)

        # Display the black-and-white mask in a separate window.
        # White areas show where the selected color was detected.
        cv2.imshow("Mask", mask)

        # KEYBOARD CONTROLS
        # waitKey(1) waits briefly for a keyboard input.
        # & 0xFF makes sure we only use the last 8 bits of the key value.
        key = cv2.waitKey(1) & 0xFF

        # Press Q to stop the main camera loop.
        if key == ord("q"):
            break

        # Press 1, 2, 3, or 4 to switch the target color.
        elif key in KEY_TO_COLOR:

            # Change the selected color based on the pressed key.
            color = KEY_TO_COLOR[key]

            # If HSV tuning is active, recreate the trackbars.
            # This updates their starting values for the new color.
            if tuning:
                cv2.destroyWindow(TUNE_WIN)
                make_trackbars(color)

        # Press T to turn HSV tuning on or off.
        elif key == ord("t"):

            # Switch the tuning setting to its opposite value.
            # False becomes True, and True becomes False.
            tuning = not tuning

            # Create the trackbars when tuning is enabled.
            # Destroy the trackbar window when tuning is disabled.
            make_trackbars(color) if tuning else cv2.destroyWindow(TUNE_WIN)

        # Press S to save a snapshot of the current camera frame and mask.
        elif key == ord("s"):

            # Create a timestamp for the filenames.
            # This helps prevent saved snapshots from overwriting each other.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the processed camera frame with its bounding boxes.
            cv2.imwrite(f"captures/{stamp}_frame.png", output)

            # Save the black-and-white detection mask.
            cv2.imwrite(f"captures/{stamp}_mask.png", mask)

            # Record the saved snapshot's filename in the SQLite database.
            store.log_snapshot(session_id, f"captures/{stamp}_frame.png")

            # Display a confirmation message in the terminal.
            print(f"[snapshot] saved captures/{stamp}_*.png")

    # SESSION CLEANUP
    # Calculate the average FPS using all recorded FPS measurements.
    # If no measurements were taken, use 0.0 to avoid division by zero.
    avg_fps = fps_sum / fps_n if fps_n else 0.0

    # Save the final session information to the database.
    # This includes the session ID, total detections, and average FPS.
    store.end_session(session_id, detection_count, avg_fps)

    # Display the final session results in the terminal.
    print(
        f"[session {session_id}] detections={detection_count} avg_fps={avg_fps:.1f}")

    # Release the camera so other applications can use it.
    cap.release()

    # Close all OpenCV windows.
    cv2.destroyAllWindows()

    # Close the SQLite database connection.
    store.close()


# This condition checks whether the file is being run directly.
# If it is, the main() function starts the color detection program.
if __name__ == "__main__":
    main()
