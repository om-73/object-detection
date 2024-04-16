import cv2
import matplotlib.pyplot as plt

# Load the pre-trained model
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn.readNetFromTensorflow(frozen_model, config_file)

# Load class labels
class_labels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

# Set preferred input size
model.setPreferableInputSize(320, 320)

# Set input scale (optional)
model.setInputScale(1.0 / 127.5)

# Set input mean (optional)
model.setInputMean((127.5, 127.5, 127.5))

# Set input swapRB (optional)
model.setInputSwapRB(True)

# Load the image
img = cv2.imread('boy.webp')

# Perform inference to get class indexes, confidences, and bounding boxes
blob = cv2.dnn.blobFromImage(img, size=(320, 320), swapRB=True, crop=False)
model.setInput(blob)
output = model.forward()

for detection in output[0, 0, :, :]:
    confidence = detection[2]
    if confidence > 0.3:  # Confidence threshold
        class_index = int(detection[1])
        class_label = class_labels[class_index]
        print("Class:", class_label, ", Confidence:", confidence)

        # Get bounding box coordinates
        x1, y1, x2, y2 = int(detection[3] * img.shape[1]), int(detection[4] * img.shape[0]), \
                         int(detection[5] * img.shape[1]), int(detection[6] * img.shape[0])

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        cv2.putText(img, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
