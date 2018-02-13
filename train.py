import numpy as np
import pandas as pd
# image prepeocessing library
from skimage import feature
from skimage.io import imread
from skimage.color import rgb2gray, rgb2yuv
from skimage.transform import resize
# Mechine learning libarary
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
# split dataset by sklearn
from sklearn.model_selection import train_test_split
# progress bar
from tqdm import tqdm
# system require
import json
import os

input_shape = ( 75, 320, 3 )

MAX_SPEED = 20
MIN_SPEED = 10


def image_preprocess( input_image, canny ):
    image = input_image[ 60 : -25, :, : ] # remove sky from the image
    if canny:
        image = rgb2gray( image )
        image = feature.canny( image, sigma = 2 )
        image = np.expand_dims( image, axis = 2 )
    image = image / 255
    return image
def random_flip( input_image, steering_angle ):
    if np.random.rand() < 0.5:
        image = np.fliplr( input_image )
        steering = -steering_angle
    else:
        image = input_image
        steering = steering_angle
    return image, steering


def load_data_3( num_of_images = 860 ):
    images = np.zeros(( num_of_images*3, 160, 320, 3 ))
    labels = np.zeros(( num_of_images*3, 1 ))

    sta = [ 'center', 'right', 'left' ]
    loop_in_sta = 0
    for k in range(3):
        _in_sta = sta[k]
        for i in range( num_of_images ):
            read_image = imread( os.path.join( data[ _in_sta ][i] ))
            images[ i + loop_in_sta ] = image_preprocess( read_image )
            labels[ i + loop_in_sta ] = data['steering'].values[i]
        loop_in_sta += num_of_images
    return images, labels

def load_data_balance( num_of_images, canny ):
    images = []
    labels = []

    sta = [ 'center', 'right', 'left' ]
    loop_in_sta = 0
    for k in range(3):
        _in_sta = sta[k]
        print(' ========== Loading Image in {} =========='.format( _in_sta ))
        for i in tqdm( range( num_of_images ) ):
            read_image = imread( os.path.join( data[ _in_sta ][i] ))
            if canny:
                read_image = image_preprocess( read_image, True )
            else:
                read_image = image_preprocess( read_image, False )
            steering_angle = data['steering'].values[i]
            throttle = data['throttle'].values[i]
            # for right and left camera argument
            if _in_sta == 'right':
                steering_angle -= 0.1
            elif _in_sta == 'left':
                steering_angle += 0.1

            # just for throttle control
            if data['speed'].values[i] < MIN_SPEED:
                throttle += 0.1

            read_image, steering_angle = random_flip( read_image, steering_angle )
            images.append( read_image )
            labels.append( [ steering_angle, throttle ] )
        
    images = np.asarray( images )
    labels = np.asarray( labels )
    return images, labels



# reprocess image path to directory
data = pd.read_csv( 'driving_log.csv', names = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
data['center'] = data['center'].str.replace( '/home/kevin/Desktop/SelfDrivingCar/', '' )
data['left'] = data['left'].str.replace( '/home/kevin/Desktop/SelfDrivingCar/', '' )
data['right'] = data['right'].str.replace( '/home/kevin/Desktop/SelfDrivingCar/', '' )

def Model():
    model = Sequential()
    model.add( Conv2D( 24, 5, 5, activation = 'relu', subsample = ( 2, 2 ), input_shape = input_shape ))
    model.add( Conv2D( 36, 5, 5, activation = 'relu', subsample = ( 2, 2 ) ) )
    model.add( Conv2D( 48, 5, 5, activation = 'relu', subsample = ( 2, 2 ) ) )
    model.add( Conv2D( 64, 3, 3, activation = 'relu' ) )
    model.add( Conv2D( 64, 3, 3, activation = 'relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Flatten() )
    model.add( Dense( 100, activation = 'relu' ) )
    model.add( Dense( 50, activation = 'relu' ) )
    model.add( Dense( 10, activation = 'relu' ) )
    model.add( Dense( 2 ) )
    model.summary()
    return model

def train( model ):
    # save the model every epoch
    checkpoint = ModelCheckpoint( '/var/www/html/custom/model-edges.h5',
                                    monitor = 'validation_loss',
                                    verbose = 0,
                                    save_best_only = True,
                                    mode = 'auto' )
    model.compile( loss = 'mean_squared_error', optimizer = Adam() )
    for i in range(100):
        print('########## LOOP NUMBER {} ##########'.format( i + 1 ))
        train_images, train_labels = load_data_balance( 1530, canny = False )
        model.fit(  train_images, train_labels,
                    batch_size = 100,
                    epochs = 10 ,
                    shuffle = True )
    model.save( '/var/www/html/custom/model-edges.h5' )


if __name__ == '__main__':
    if os.path.isfile('model-best.h5'):
        print('---------- Loading Model ----------')
        model = load_model('model-best.h5')
    else:
        model = Model()
    train( model )
'''
    with open('model-edges.json', 'w' ) as json_file:
        json.dump( model.to_json(), json_file )
'''


