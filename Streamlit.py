import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()


st.title('Infection Detection')

st.markdown("Welcome to this simple web application that uses an image classifier to identify infection in wounds")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
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
    classifier_model = '/Users/kevinmcdonough/Documents/Flatiron/capstone/project/Capstone/weights.best.h5'
    model = load_model(classifier_model)
    test_image = image.resize((300,300))
    img_array = np.array(test_image).astype('float32')/255
    img_array = img_array.reshape(300,300,3)
    img_array = np.expand_dims(img_array, axis=0)
    class_names = [
          'Infection',
          'No Infection']
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Infection': 0,
          'No Infection': 0,
}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result

if __name__ == "__main__":
    main()