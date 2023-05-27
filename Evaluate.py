
import numpy as np
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model architecture from JSON file
with open('model/emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)

# Load the model weights
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Set up the test data generator
test_data_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_data_gen.flow_from_directory(
    'test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# Make predictions on the test data
predictions = emotion_model.predict_generator(test_generator)

# Compute confusion matrix
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))

# Plot confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Print classification report
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))








