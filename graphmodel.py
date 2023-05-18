import cv2

# Load the frozen graph (.pb)
net = cv2.dnn.readNetFromTensorflow('/home/roman/Desktop/proiect/graph/frozen_graph.pb')

# Load and preprocess the image
img = cv2.imread('/home/roman/Desktop/proiect')
blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Perform object detection
detections = net.forward()

# Process the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Adjust confidence threshold as needed
        class_id = int(detections[0, 0, i, 1])
        if class_id == 7:  # Assuming class ID 7 corresponds to cans
            box = detections[0, 0, i, 3:7] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.putText(img, 'can', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
