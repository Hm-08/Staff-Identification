from ultralytics import YOLO
import cv2

# Load the custom model
model = YOLO('best.pt')

# Define video paths
video_path = 'sample.mp4'
video_path_out = '{}_out.mp4'.format(video_path)

# Load the video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Open a text file to write the frames and coordinates
output_file = open('staff_coordinates.txt', 'w')

frame_id = 0
frames_with_staff = []
threshold = 0.5

while ret:

    results = model(frame)[0]

    staff_count = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            if int(class_id) == 0: # class_id (0: staff, 1: staff_tag)
                staff_count += 1
                center_x = int(x1 + x2) // 2
                center_y = int(y1 + y2) // 2
                frames_with_staff.append((frame_id, (center_x, center_y)))

                # Write to the text file
                output_file.write(f"Frame: {frame_id}, Coordinates: ({center_x}, {center_y})\n")
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    text = f'Frame: {frame_id}, Detected Staff: {staff_count}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Staff Identification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

# Print the frames with staff and their coordinates
print("Frames with staff present and their coordinates:")
for frame_data in frames_with_staff:
    print(f"Frame: {frame_data[0]}, Coordinates: {frame_data[1]}")
