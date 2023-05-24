 !pip install rasterio
 
import tensorflow as tf
from google.colab import drive
import os
import numpy as np
import cv2
import rasterio
from rasterio.plot import show

 # Set Google Drive
drive.mount('/content/drive')
 
# Image folder
folder_path_nbr_values = '/content/drive/MyDrive/CLASEML/FINAL/'

#Image files
file_list = os.listdir(folder_path_nbr_values)
image_files = [file for file in file_list if file.endswith(('.jpg', '.png', '.tif', '.bmp'))]
num_imagenes = len(image_files)
print(f"La carpeta contiene {num_imagenes} imágenes.")

 #  Lists to store input and target images
input_images = []
target_images = []
 
#  Lists to store input and target images
for image_file in image_files:
    image_path = os.path.join(folder_path_nbr_values, image_file)
 
with rasterio.open(image_path) as dataset:
    # Get the image matrix
    image_data = dataset.read()
    
    # Get NBR and NDVI values
    nbr_values = image_data[0, :, :]  # Band 1 (NBR)
    ndvi_values = image_data[1, :, :]  # Band 2 (NDVI)
    
    # Remove NaN values from the arrays
    nbr_values = np.nan_to_num(nbr_values)
    ndvi_values = np.nan_to_num(ndvi_values)
    
    # Add input and target images
    input_images.append([nbr_values, ndvi_values])
    target_images.append(nbr_values)  # Predict NBR values
    
    # Check NBR and NDVI values
    print("NBR values:")
    print(nbr_values)
    print("NDVI values:")
    print(ndvi_values)

    # Visualize the image (optional)
    # show(nbr_values, cmap='gray')
    # show(ndvi_values, cmap='gray')

#  Check image shapes
input_shapes = set(np.array(image).shape for image in input_images)
target_shapes = set(np.array(image).shape for image in target_images)

 #  Ensure all images have the same shape
if len(input_shapes) != 1 or len(target_shapes) != 1:
 print("Las imágenes no tienen la misma forma")
else:
 #  Ensure all images have the same shape
    common_shape = input_shapes.pop()

 #  Convert image lists to numpy arrays 
    input_images = np.stack([np.reshape(np.array(image), common_shape) for image in input_images])
    target_images = np.stack([np.reshape(np.array(image), common_shape) for image in target_images])

 input_shapes = set(np.array(image).shape for image in input_images)
target_shapes = set(np.array(image).shape for image in target_images)

print("Shapes of input images:")
print(input_shapes)
print("Shapes of target images:")
print(target_shapes)
 #  Get the total size of input images
input_total_size = np.prod(input_shape_desired)

#  Ensure images have the correct total size
if input_images_reshaped.size != input_total_size:
    input_images_reshaped = tf.image.resize(input_images_reshaped, (input_shape_desired[1], input_shape_desired[2]), method=tf.image.ResizeMethod.BILINEAR).numpy()

#  Reshape input images
input_images_reshaped = input_images_reshaped.reshape(-1, input_shape_desired[0], input_shape_desired[1], input_shape_desired[2])

#  Reshape target images
target_images_reshaped = target_images_reshaped.reshape(-1, target_shape_desired[0], target_shape_desired[1])

 # Get desired shapes
input_shape_desired = (2, 48, 39)  #  Desired shape of input images 
target_shape_desired = (48, 39)  #   Desired shape of target images 

 #  Reshape input images
input_images_reshaped = []
for image in input_images:
 if image[0].shape == input_shape_desired:
        input_images_reshaped.append(image)
 else:
        reshaped_image = cv2.resize(image[0], (input_shape_desired[1], input_shape_desired[0]), interpolation=cv2.INTER_LINEAR)
        input_images_reshaped.append(reshaped_image)

#  Convert list to numpy array
input_images_reshaped = np.stack(input_images_reshaped)

# Reshape target images
target_images_reshaped = []
for image in target_images:
 if image.shape == target_shape_desired:
        target_images_reshaped.append(image)
 else:
        reshaped_image = cv2.resize(image, (target_shape_desired[1], target_shape_desired[0]), interpolation=cv2.INTER_LINEAR)
        target_images_reshaped.append(reshaped_image)

#  Convert list to numpy array
target_images_reshaped = np.stack(target_images_reshaped)

#  Ensure images have the correct dimensions
input_images_reshaped = input_images_reshaped.reshape(-1, input_shape_desired[0], input_shape_desired[1], 1)
target_images_reshaped = target_images_reshaped.reshape(-1, target_shape_desired[0], target_shape_desired[1], 1)

 #  Check reshaped image shapes
input_reshaped_shapes = set(np.array(image).shape for image in input_images_reshaped)
target_reshaped_shapes = set(np.array(image).shape for image in target_images_reshaped)

print("Reshaped input image shapes:")
print(input_reshaped_shapes)
print("Reshaped target image shapes:")
print(target_reshaped_shapes)

#  Check if all reshaped images have the same shape
if len(input_reshaped_shapes) != 1 or len(target_reshaped_shapes) != 1:
 print(" Reshaped images do not have the same shape ")
else:
 print(" All reshaped images have the same shape ")

 #  Define the Encoder-Decoder model
encoder_nbr_input = tf.keras.layers.Input(shape=(None, None, 1))
encoder_nbr = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(encoder_nbr_input)
encoder_nbr = tf.keras.layers.MaxPooling2D((2, 2))(encoder_nbr)
encoder_nbr = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(encoder_nbr)
encoder_nbr = tf.keras.layers.MaxPooling2D((2, 2))(encoder_nbr)

encoder_ndvi_input = tf.keras.layers.Input(shape=(None, None, 1))
encoder_ndvi = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(encoder_ndvi_input)
encoder_ndvi = tf.keras.layers.MaxPooling2D((2, 2))(encoder_ndvi)
encoder_ndvi = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(encoder_ndvi)
encoder_ndvi = tf.keras.layers.MaxPooling2D((2, 2))(encoder_ndvi)

 #  Combine encoder outputs
encoder_combined = tf.keras.layers.concatenate([encoder_nbr, encoder_ndvi])

decoder = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(encoder_combined)
decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)
decoder = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(decoder)
decoder = tf.keras.layers.UpSampling2D((2, 2))(decoder)
decoder = tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(decoder)

model = tf.keras.Model(inputs=[encoder_nbr_input, encoder_ndvi_input], outputs=decoder)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 #  Train the model
model.fit(input_images_reshaped.reshape(-1, 48, 39), target_images.reshape(-1, 48, 39), epochs=10, batch_size=16)