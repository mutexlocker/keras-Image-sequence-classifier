import numpy as np
from keras import backend as K
import sys
import os


def main():
    testsetCount = 5
    accuracies = [0, 0, 0,0,0]

    for testid in range(0,testsetCount):
        print(testid)
        K.set_image_dim_ordering('tf')
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
        from keras_video_classifier.library.utility.plot_utils import plot_and_save_history
        from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf

        data_set_name = 'AM_pics_' + str(testid)
        input_dir_path = os.path.join(os.path.dirname(__file__), 'AM_data')
        output_dir_path = os.path.join(os.path.dirname(__file__), 'models', data_set_name)
        report_dir_path = os.path.join(os.path.dirname(__file__), 'reports', data_set_name)

        np.random.seed(42)

        # this line downloads the video files of UCF-101 dataset if they are not available in the very_large_data folder
        #load_ucf(input_dir_path)

        classifier = VGG16BidirectionalLSTMVideoClassifier()

        history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, vgg16_include_top=False,
                                 data_set_name=data_set_name, from_picture=True)

        plot_and_save_history(history, VGG16BidirectionalLSTMVideoClassifier.model_name,
                              report_dir_path + '/' + VGG16BidirectionalLSTMVideoClassifier.model_name + '-hi-dim-history.png')


        print("predicting now !!!!!")
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
        from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

        vgg16_include_top = False
        data_dir_path = os.path.join(os.path.dirname(__file__), 'AM_data')
        model_dir_path = os.path.join(os.path.dirname(__file__), 'models', data_set_name)

        config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                      vgg16_include_top=vgg16_include_top)
        weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                      vgg16_include_top=vgg16_include_top)
        print("Reading weights from :", weight_file_path)
        print("Reading Config from :", config_file_path)
        np.random.seed(33)

        #load_ucf(data_dir_path)

        predictor = VGG16BidirectionalLSTMVideoClassifier()
        predictor.load_model(config_file_path, weight_file_path)

        videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()],testid)

        video_file_path_list = np.array([file_path for file_path in videos.keys()])
        np.random.shuffle(video_file_path_list)

        correct_count = 0
        count = 0

        for video_file_path in video_file_path_list:
            label = videos[video_file_path]
            predicted_label = predictor.predict(video_file_path,from_picture=True)
            print('predicted: ' + predicted_label + ' actual: ' + label)
            correct_count = correct_count + 1 if label == predicted_label else correct_count
            count += 1
            accuracy = correct_count / count
            print('accuracy: ', accuracy)
            accuracies[testid] = accuracy

    print(accuracies)
    print(sum(accuracies) / len(accuracies))



if __name__ == '__main__':
    main()
