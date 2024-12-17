import io
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

def create_model():
    input_layer = tf.keras.Input(shape=(28, 28, 1))
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='Conv2D_1')(input_layer)
    x = layers.MaxPooling2D((2, 2), name='MaxPooling2D_1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv2D_2')(x)
    x = layers.MaxPooling2D((2, 2), name='MaxPooling2D_2')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', name='Dense_1')(x)
    x = layers.Dense(32, activation='relu', name='Dense_2')(x)
    output_layer = layers.Dense(10, activation='softmax', name='Output')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

model = create_model()
layer_outputs = [layer.output for layer in model.layers]
intermediate_model = models.Model(inputs=model.input, outputs=layer_outputs)

def preprocess_image(img):
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    img_array = img_array / 255.0
    return img_array

def add_color_to_feature_map(feature_map):
    colored_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    return colored_map

def plot_feature_maps(feature_maps):
    max_display_size = 450  # Increased from 300 to 450
    margin = 20  # Increased margin
    node_margin = 2
    width_needed = (max_display_size + margin) * len(model.layers) + margin
    height_needed = max_display_size + 60  # Increased to provide more space

    output_img = np.ones((height_needed, width_needed, 3), dtype=np.uint8) * 0  # Black background

    y_offset = 10

    for idx, layer in enumerate(model.layers):
        features = feature_maps[idx]
        layer_name = layer.name

        if len(features.shape) == 4:
            num_filters = features.shape[-1]
            size = features.shape[1]
            grid_cols = int(np.ceil(np.sqrt(num_filters)))
            grid_rows = int(np.ceil(num_filters / grid_cols))
            canvas_size = size + node_margin
            display_grid = np.zeros((canvas_size * grid_rows, canvas_size * grid_cols), dtype=np.uint8)

            for i in range(num_filters):
                x = features[0, :, :, i]
                x -= x.mean()
                x /= (x.std() + 1e-5)
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')

                row = i // grid_cols
                col = i % grid_cols
                display_grid[row * canvas_size:(row * canvas_size + size), col * canvas_size:(col * canvas_size + size)] = x

            display_grid = add_color_to_feature_map(display_grid)
            display_grid = cv2.resize(display_grid, (max_display_size, max_display_size), interpolation=cv2.INTER_NEAREST)

            x_offset = margin + (idx * (max_display_size + margin))
            output_img[y_offset:y_offset + max_display_size, x_offset:x_offset + max_display_size] = display_grid

            cv2.putText(output_img, layer_name, (x_offset, y_offset + max_display_size + 40),  # Increased position offset
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)  # Increased font scale and thickness

        elif len(features.shape) == 2:
            fig = plt.figure(figsize=(2.5, 2.5))  # Increased figure size
            plt.bar(range(features.shape[1]), features[0])
            plt.ylim(0, 1)
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            bar_img = np.array(Image.open(buf))
            bar_img = cv2.resize(bar_img, (max_display_size, max_display_size))

            x_offset = margin + (idx * (max_display_size + margin))
            bar_img = cv2.cvtColor(bar_img, cv2.COLOR_RGBA2BGR)
            h, w = bar_img.shape[:2]
            output_img[y_offset:y_offset + h, x_offset:x_offset + w] = bar_img

            cv2.putText(output_img, layer_name, (x_offset, y_offset + h + 40),  # Increased position offset
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)  # Increased font scale and thickness

    return output_img

st.title("Drawable Canvas")
st.markdown("""
Draw on the canvas and see the feature maps processed through the CNN!
""")

st.sidebar.header("Configuration")

b_width = st.sidebar.slider("Brush width: ", 1, 100, 10)
b_color = st.sidebar.color_picker("Enter brush color: ")
bg_color = st.sidebar.color_picker("Enter background color: ", "#FFFFFF")
drawing_mode = st.sidebar.checkbox("Drawing mode?", True)

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=b_width,
    stroke_color=b_color,
    background_color=bg_color,
    height=200, 
    width=200, 
    drawing_mode="freedraw" if drawing_mode else "transform",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray(np.uint8(img)).convert("RGB")
    preprocessed_img = preprocess_image(img)
    feature_maps = intermediate_model.predict(preprocessed_img)

    output_img = plot_feature_maps(feature_maps)
    st.image(output_img, caption='Feature Maps')