#!/home/kevin/Desktop/SelfDrivingCar/keras/bin/python
import base64
from datetime import datetime
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

# image prepeocessing library
from skimage import feature
from skimage.color import rgb2gray, rgb2yuv
from skimage.transform import resize

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#set gpu memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

file_name = 'model-edges.h5'
# check file model-edges.h5 exist
import os
import urllib

if True:
    if os.path.isfile( file_name ) == False:
        print('====== Downloading file ======')
        urllib.request.urlretrieve("http://192.168.1.25/custom/model-edges.h5", file_name )

socket = socketio.Server()
app = Flask(__name__)

model = load_model( file_name )
prev_image = None

Max_speed = 25
Min_speed = 10

datagen = ImageDataGenerator(
    rescale = 1./ 255,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False )  # randomly flip images

def send_control( steering_angle, throttle ):
    socket.emit(
        "steer",
        data = {
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid = True )

def transfer_image( image ):
    tmp_image = rgb2gray( image )
    tmp_image = feature.canny( tmp_image, sigma = 3 )
    # ( 160, 320 ) -> ( 160, 320, 1 )
    tmp_image = np.expand_dims( tmp_image / 255, axis = 2 )
    # ( 160, 320, 1 ) -> ( 1, 160, 320, 1 )
    tmp_image = np.expand_dims( tmp_image, axis = 0 )
    return tmp_image

def transfer_image_normal( image ):
    tmp_image = np.expand_dims( image / 255, axis = 0 )
    return tmp_image

def image_preprocess( input_image ):
    image = input_image[ 60 : -25, :, : ] # remove sky from the image
    image = image / 255
    image = np.expand_dims( image, axis = 0 )
    return image


@socket.on('telemetry')
def telemetry( sid, data ):
    if data:
        steering_angle = float( data['steering_angle'] )
        throttle = float( data['throttle'] )
        speed = float( data['speed'] )
        # the center of the camera
        image = Image.open( BytesIO( base64.b64decode( data['image'] ) ) )
        try:
            image = np.asarray( image )
            image_predict = image_preprocess( image )

            output = model.predict( image_predict )
            output = output[0]
            steering_angle = float( output[0] )
            throttle = float( output[1] )

            if speed < 15:
                throttle = 0.5

            send_control( steering_angle, throttle )
            print( '{} {} {}'.format( steering_angle , throttle, speed ) )
        except Exception as e:
            print(e)

@socket.on('connect')
def connect( sid, environ ):
    print( 'connect -> {}'.format( sid ) )

if __name__ == '__main__':
    app = socketio.Middleware( socket, app )
    eventlet.wsgi.server( eventlet.listen(( '', 4567 )), app )


