import cv2

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# Initialize camera
cap = cv2.VideoCapture(0)

window_width = 1200
window_height = 923

# Set the window properties
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", window_width, window_height)

while True:
    # Get frames
    ret, frame = cap.read()

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)

    # Initialize object count dictionary
    object_counts = {}

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
        class_name = classes[class_id]
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)

        # Update object count
        if class_name not in object_counts:
            object_counts[class_name] = 1
        else:
            object_counts[class_name] += 1

    # Display object counts
    for object_name, count in object_counts.items():
        cv2.putText(frame, f"{object_name}: {count}", (10, 30 * (list(object_counts.keys()).index(object_name) + 1)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()