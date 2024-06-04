from math import ceil
import numpy as np

def predict_image_class(model, image_array, class_names):
    # Predict class probabilities
    predictions = model.predict(image_array)
    # Get the class with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Get the confidence score of the predicted class
    confidence_score = predictions[0][predicted_class_index] * 100

    # Round the confidence score up
    rounded_confidence_score = ceil(confidence_score)

    print("confidence :", rounded_confidence_score)
    # Return the class name and confidence score
    return class_names[predicted_class_index], rounded_confidence_score