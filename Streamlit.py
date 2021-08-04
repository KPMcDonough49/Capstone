import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import load_img
import time
fig = plt.figure()


st.title('Infection Detection')

st.markdown("Welcome to this simple web application that uses an image classifier to detect infection in wounds. This app is meant to serve as a screening tool.")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                

def predict(image):
    classifier_model = 'my_model.h5'
    model = load_model(classifier_model)
    test_image = image.resize((300,300))
    img_array = np.array(test_image).astype('float32')/255
    img_array = img_array.reshape(300,300,3)
    img_array = np.expand_dims(img_array, axis=0)
    class_names = [
          'Infection',
          'No Infection']
    predictions = model.predict(img_array)
    classification = np.argmax(predictions, axis=-1)
    class_index = int(classification)
    pred_percentage = predictions[0][class_index]
    if class_names[class_index] == "No Infection":
        message = " Continue to monitor and set up an appointment if it worsens."
    else:
        message = " Recommend seeing the Doctor."
    result = f"{class_names[class_index]} with a { (100 * pred_percentage).round(2) } % confidence." + message

    return result

if __name__ == "__main__":
    main()