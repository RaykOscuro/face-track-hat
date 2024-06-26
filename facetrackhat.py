import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(1)  # 0 for default camera

# Load the hat image
hat_image = cv2.imread("hat.png", -1)

# Initialize the previous face coordinates
prev_x = 0
prev_y = 0
prev_w = 0
prev_h = 0

# Capture the first frame
prevFrame = cap.read()[1]
height, width, _ = prevFrame.shape
height = height + 300

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        continue

    large_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Copy the original frame into the new frame, starting from the bottom
    large_frame[300:, :, :] = frame

    # Use the new frame from now on
    frame = large_frame
    b, g, r, a = cv2.split(hat_image)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.erode(a, None, iterations=5)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cv2.CascadeClassifier("haarcascade_profileface.xml").detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    # Detect faces in the frame
    if len(faces) == 0:
        faces = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        # Adjust the y-coordinate to display the hat higher on the face
        y -= int(h * 0.8)  # Shift the hat position by 80% of the face height

        # Ensure the y-coordinate does not go out of bounds
        if y < 0:
            y = 0
        # Crop the face region
        frame = np.full(
            (int(frame.shape[0]), frame.shape[1], 3), (0, 255, 0), dtype=np.uint8
        )
        h = (prev_h * 3 + h) // 4 + 20
        w = (prev_w * 3 + w) // 4 + 20
        prev_h = h - 20
        prev_w = w - 20
        face_region = frame[y : y + h, x : x + w]

        # Resize the hat image to match the face region
        hat_resized = cv2.resize(
            overlay_color,
            (face_region.shape[1], face_region.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (face_region.shape[1], face_region.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        # Overlay the hat image on the face region
        print(w, h)
        print(face_region.shape, mask.shape)
        combined_image_bg = cv2.bitwise_and(
            face_region, face_region, mask=cv2.bitwise_not(mask)
        )
        combined_image_fg = cv2.bitwise_and(hat_resized, hat_resized, mask=mask)
        x = (x + prev_x) // 2
        y = (y + prev_y) // 2
        prev_x = x
        prev_y = y
        frame_space = large_frame[
            y : y + combined_image_bg.shape[0], x : x + combined_image_bg.shape[1]
        ].shape
        if frame_space != combined_image_bg.shape:
            combined_image_bg.resize(frame_space)
            combined_image_fg.resize(frame_space)
        frame[
            y : y + combined_image_bg.shape[0], x : x + combined_image_bg.shape[1]
        ] = cv2.add(combined_image_bg, combined_image_fg)
        prevFrame = frame
    # Display the frame with the tracked face and added hat
    if len(faces) == 0:
        frame = prevFrame
    cv2.imshow("Face with Hat", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
