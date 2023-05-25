import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

emotion_dict = {0: "Angry(negative)", 1: "Disgusted(negative)", 2: "Fearful(negative)", 3: "Happy(positive)", 4: "Neutral", 5: "Sad(negative)", 6: "Surprised(neutral)"}

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

sentiments={"positive": 0, "negative": 0, "neutral": 0}

# Specify the video path and start the capture
video_path = "C:\\Project\\ML\\Sample_videos\\two.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successfully opened
if not cap.isOpened():
    print("Error opening video file:", video_path)
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        if(maxindex in (0,1,2,5)):
            sentiments["negative"] += 1
        elif(maxindex in(4,6)):
            sentiments["neutral"] += 1
        else:
            sentiments["positive"] += 1
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

labels = list(sentiments.keys())
sizes = list(sentiments.values())
colors = ['green', 'red', 'gray']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Sentiment Analysis Results')

# Display the pie chart
plt.show()


