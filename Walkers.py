import cv2

# Path to the full body Haar cascade classifier file
body_cascade_file = "haarcascade_fullbody.xml"

# Load the body classifier
body_classifier = cv2.CascadeClassifier(body_cascade_file)

# Open the video capture stream
cap = cv2.VideoCapture("walking.avi")  # Change to video file path if needed

while True:
  # Read a frame from the video
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
    print("Error: Could not read frame")
    break

  # Convert the frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect full bodies in the grayscale frame
  bodies = body_classifier.detectMultiScale(gray, 1.1, 4)

  # Draw rectangles around detected bodies
  for (x, y, w, h) in bodies:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Red rectangle

  # Display the frame with detected bodies
  cv2.imshow("Full Body Detection", frame)

  # Exit if 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
