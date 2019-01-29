import cv2
import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
import pylab
MAX_NB_CLASSES = 20



def extract_vgg16_features_live(model, video_input_file_path):
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    return unscaled_features


def extract_vgg16_features_live_pictures(model, picture_input_file_path):
    print('Extracting features from pictures: ', picture_input_file_path)
    filenames = os.listdir(picture_input_file_path)
    filenames.sort(key=lambda x: int(x[:-4]))
    features = []
    for filename in filenames:
        image_full_path = picture_input_file_path + '/' + filename
        image = cv2.imread(image_full_path)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features.append(feature)
    unscaled_features = np.array(features)
    return unscaled_features


def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from videoo: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            #cv2.imwrite(str('tmpx' + str(count) + '.png'), img)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def extract_vgg16_features_pictures(model, picture_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    filenames = os.listdir(picture_input_file_path)
    filenames.sort(key=lambda x: int(x[:-4]))
    features = []
    for filename in filenames:
        image_full_path = picture_input_file_path+'/'+filename
        print(image_full_path)
        image = cv2.imread(image_full_path)
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = model.predict(input).ravel()
        features.append(feature)
        ### virtulize extracted features
        #jiz = model.predict(input)
        #pic = jiz[0, :, :, 1]
        #pylab.imshow(pic)
        #pylab.gray()
        #pylab.show()
    unscaled_features = np.array(features)
    print('Saving features to  :', feature_output_file_path)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features



def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None, is_picture=False):
    if data_set_name is None:
        data_set_name = 'UCF-101'
    if is_picture:
        input_data_dir_path = data_dir_path + '/' + data_set_name
    else:
        input_data_dir_path = data_dir_path + '/' + data_set_name
    print("Scan and extract from this path : ", input_data_dir_path)
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

    if model is None:
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                if is_picture:
                    print('Extract features from :', video_file_path)
                    x = extract_vgg16_features_pictures(model, video_file_path, output_feature_file_path)
                else:
                    x = extract_vgg16_features(model, video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

