import tensorflow.keras as tk
import numpy as np
from PIL import Image
import streamlit as st

model = tk.models.load_model('MNIST_classifier_nn_model.h5')
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Shoe']

st.title("Fashion Product Prediction")
file = st.file_uploader("Choose a file", type=['png', 'jpg'])

st.write("Instruction of Image")
st.write("\t1. Background Of Image should be White.")


if file is not None:
    img = Image.open(file)
    # st.image(file)
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            r = 255 - r
            g = 255 - g
            b = 255 - b
            img.putpixel((i, j), (r, g, b))

    img = img.convert('L')  # //Convert to grayScale.

    img = img.resize((28, 28), Image.ANTIALIAS)  # //convert image into 28x28.

    numpydata = np.asarray(img)

    numpydata = numpydata.reshape(-1, 28, 28)  # //convert image.
    numpydata = numpydata / 255

    y_pred = model.predict(numpydata)
    pred_idx = np.argmax(y_pred[0])

    col1, col2 = st.columns(2)
    with col1:
        st.image(file,use_column_width=True)
    with col2:
        st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Predicted Product"}</h1>', unsafe_allow_html=True)
        st.title(class_labels[pred_idx])  # predicted Value


else:
    st.text("Please upload an image file")

