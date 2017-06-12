import numpy as np
import cv2
import skvideo.io
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from os.path import join, exists, splitext

def convert_color(imgs, color_space):
    """
    Converts RGB images to the given color space.
    :param imgs: The RGB images to convert.
    :param color_space: The color space to which to convert the images.
                        Options: Gray, HSV, LUV, HLS, YUV, YCrCb
    :return: The color-converted versions of imgs.
    """
    assert color_space in ['Gray', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], \
        "Color space must be one of 'Gray', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'"

    imgs_converted = np.empty_like(imgs)

    # Convert every image in imgs.
    for i, img in enumerate(imgs):
        if color_space == 'Gray':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == 'HSV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        else: # color_space == 'YCrCb':
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

        imgs_converted[i] = img_converted

    return imgs_converted


def get_HOG_features(imgs, num_orientations=c.NUM_ORIENTATIONS, pix_per_cell=c.PIX_PER_CELL,
                     cells_per_block=c.CELLS_PER_BLOCK, feature_vec=True, visualize=False):
    """
    Calculates the Histogram of Oriented Gradient features for the relevant region (lower half) of
    the given images.
    :param imgs: The images for which to calculate HOG features.
    :param num_orientations: The number of gradient orientation bins for the histogram.
    :param pix_per_cell: The number of pixels in a HOG cell.
    :param cells_per_block: The number of HOG cells in a block (for block normalization).
    :param feature_vec: Whether to return as a 1D array of features (True) or keep the dimensions of
                        imgs (False).
    :param visualize: Whether to return a tuple, (features, visualization img), (True) or just the
                      features (False).
    :return: The HOG features for imgs.
    """
    features = []  # Easier to use lists than np arrays because dimensions vary based on inputs.

    if visualize:
        hog_imgs = []

        for img in imgs:
            c1_features, c1_img = hog(img[:, :, 0],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)
            c2_features, c2_img = hog(img[:, :, 1],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)
            c3_features, c3_img = hog(img[:, :, 2],
                                      orientations=num_orientations,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cells_per_block, cells_per_block),
                                      transform_sqrt=True,
                                      visualise=visualize,
                                      feature_vector=feature_vec)

            if feature_vec:
                features.append(np.concatenate([c1_features,
                                                c2_features,
                                                c3_features]))
                hog_imgs.append(np.concatenate([c1_img,
                                                c2_img,
                                                c3_img]))
            else:
                features.append(np.array([c1_features,
                                          c2_features,
                                          c3_features]))
                hog_imgs.append(np.array([c1_img,
                                          c2_img,
                                          c3_img]))

        return np.array(features), hog_imgs
    else:
        for img in imgs:
            c1_features = hog(img[:,:,0],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)
            c2_features = hog(img[:,:,1],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)
            c3_features = hog(img[:,:,2],
                              orientations=num_orientations,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block),
                              transform_sqrt=True,
                              visualise=visualize,
                              feature_vector=feature_vec)

            if feature_vec:
                features.append(np.concatenate([c1_features,
                                                c2_features,
                                                c3_features]))
            else:
                features.append(np.array([c1_features,
                                          c2_features,
                                          c3_features]))

        return np.array(features)

def get_color_bin_features(imgs, shape=c.COLOR_BIN_SHAPE):
    """
    Calculates color bin features for the given images by downsizing and taking each pixel as
    representative of the colors of the surrounding pixels in the full-size image.
    :param imgs: The images for which to calculate color bin features.
    :param shape: A tuple, (height, width) - the shape to which imgs should be downsized.
    :return: The color bin features for imgs.
    """
    # Sized to hold the ravelled pixels of each downsized image.
    features = np.empty([imgs.shape[0], shape[0] * shape[1] * imgs.shape[3]])

    # Resize and ravel every image to get color bin features.
    for i, img in enumerate(imgs):
        features[i] = cv2.resize(img, shape).ravel()

    return features

def get_color_hist_features(imgs, nbins=c.NUM_HIST_BINS, bins_range=c.HIST_BINS_RANGE):
    """
    Calculates color histogram features for each channel of the given images.
    :param imgs: The images for which to calculate a color histogram.
    :param nbins: The number of histogram bins to sort the color values into.
    :param bins_range: The range of values over all bins.
    :return: The color histogram features of each channel for every image in imgs.
    """
    num_features = imgs.shape[-1] * nbins
    hist_features = np.empty([len(imgs), num_features])

    for i, img in enumerate(imgs):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features[i] = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


def read_input(paths):
    """
    Reads images from a list of input paths into a numpy array. Paths can either be .png for single
    images or .mp4 for videos.
    :param paths: The list of paths to read.
    :return: A numpy array of images - the frames from each path in paths, concatenated.
    """
    frames = []

    for path in paths:
        ext = splitext(path)[1]
        assert ext == '.png' or ext == '.mp4', 'The input file must be a .png or .mp4.'

        if ext == '.png':
            # Input is a single image.
            img = cv2.imread(path)
            # turn into a 4D array so all functions can apply to images and video.
            frames.append(np.array([img]))
        else:
            # Input is a video.
            vidcap = cv2.VideoCapture(path)

            # Load frames
            frames_list = []
            while vidcap.isOpened():
                ret, frame = vidcap.read()

                if ret:
                    frames_list.append(frame)
                else:
                    break

            vidcap.release()

            frames.append(np.array(frames_list))

    return np.concatenate(frames)


def save_output(imgs, path):
    """
    Saves imgs to file. Paths can either be .png for single images or .mp4 for videos.
    :param imgs: The frames to save. A single image for .pngs, or multiple frames for .mp4s.
    :param path: The path to which the image / video will be saved.
    """
    ext = splitext(path)[1]
    assert ext == '.png' or ext == '.mp4', 'The output file must be a .png or .mp4.'

    if ext == '.png':
        # Output is a single image.
        cv2.imwrite(path, imgs[0])
    else:
        # Output is a video.
        vid_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        skvideo.io.vwrite(path, vid_frames)


def save_model(model, scaler, path=c.MODEL_SAVE_PATH):
    """
    Saves a trained model to file.
    :param model: The trained scikit learn model.
    :param scaler: The scaler used to normalize the data when training.
    :param path: The filepath to which to save the model.
    """
    save_dict = {'model': model, 'scaler':scaler}
    joblib.dump(save_dict, path)


def load_model(path=c.MODEL_SAVE_PATH):
    """
    Loads a trained model from file.
    :param path: The filepath from which to load the model.
    :return: A tuple, (model, scaler).
    """
    save_dict = joblib.load(path)
    print 'Model loaded from %s' % path

    model = save_dict['model']
    scaler = save_dict['scaler']

    return model, scaler


def get_feature_vectors(imgs, color_space=c.COLOR_SPACE):
    """
    Gets the feature vectors for the given images. Used to featurize training images.
    :param imgs: The images for which to get feature vectors.
    :param color_space: The color space to which to convert the images.
    :return: The feature vectors (with HOG and color features) for imgs.
    """
    imgs_cvt = convert_color(imgs, color_space)

    HOG_features = get_HOG_features(imgs_cvt)
    color_bin_features = get_color_bin_features(imgs_cvt)
    color_hist_features = get_color_hist_features(imgs_cvt)

    # Concatenate feature vectors for each image.
    features = np.concatenate([HOG_features, color_hist_features, color_bin_features], axis=1)

    return features


def get_train_test_data(train_frac=0.66):
    """
    Loads the train and test images from file, shuffles and splits them into train and test sets.
    :param train_frac: The percentage of images to use as training data.
                       (The rest will be testing data).
    :return: A tuple of tuples, ((images train, labels train), (images test, labels test)).
    """
    
    # Split on the directories to eliminate train/test bleed
    car_paths_train = glob(join(c.DATA_DIR, 'vehicles', 'GTI*', '*.png'))
    car_paths_test = glob(join(c.DATA_DIR, 'vehicles', 'KITTI*', '*.png'))
    non_paths_train = glob(join(c.DATA_DIR, 'non-vehicles', 'GTI*', '*.png'))
    non_paths_test = glob(join(c.DATA_DIR, 'non-vehicles', 'Extras', '*.png'))

    print 'Read Input'
    car_imgs_train = read_input(car_paths_train)
    car_imgs_test = read_input(car_paths_test)
    non_imgs_train = read_input(non_paths_train)
    non_imgs_test = read_input(non_paths_test)

    imgs_train = np.concatenate([car_imgs_train, non_imgs_train])
    imgs_test = np.concatenate([car_imgs_test, non_imgs_test])

    labels_train = np.concatenate([np.ones([len(car_imgs_train)]), np.zeros([len(non_imgs_train)])])
    labels_test = np.concatenate([np.ones([len(car_imgs_test)]), np.zeros([len(non_imgs_test)])])

    print 'Get Features'
    inputs_train = get_feature_vectors(imgs_train)
    inputs_test = get_feature_vectors(imgs_test)

    inputs_train, labels_train = zip(*np.random.permutation(zip(inputs_train, labels_train)))
    inputs_test, labels_test = zip(*np.random.permutation(zip(inputs_test, labels_test)))

    # Normalize the features
    inputs = np.concatenate([inputs_train, inputs_test])
    scaler = StandardScaler().fit(inputs)

    inputs_train_norm = scaler.transform(inputs_train)
    inputs_test_norm = scaler.transform(inputs_test)

    return (inputs_train_norm, labels_train), (inputs_test_norm, labels_test), scaler


def train_model(load=True, load_path=c.MODEL_SAVE_PATH, save=True, save_path=c.MODEL_SAVE_PATH):
    """
    Returns a trained model. Trains a new model if load = False or no saved model exists. Otherwise,
    loads and returns the saved model from file.
    :param load: Whether to load a previously-trained model from load_path.
    :param load_path: The path from which to load a trained model if load = True.
    :param save: Whether to save the trained model to save_path.
    :param save_path: The path to which to save the trained model if save - True.
    :return: A model trained to classify car images vs non-car images.
    """
    # If there is a previously trained model and we want to use that, load and return it.
    if load and exists(load_path):
        print 'Loading pretrained model...'
        return load_model(path=load_path)

    # Otherwise, train a new model.
    print 'Get Data'
    (inputs_train, labels_train), (inputs_test, labels_test), scaler = get_train_test_data()

    # print 'Train Model'
    # model = LinearSVC()
    # model.fit(inputs_train, labels_train)
    #
    # # Test the model.
    # test_score = model.score(inputs_test, labels_test)
    # print 'Test Accuracy: ', test_score

    print 'Full Train'
    # Train on all the data
    inputs = np.concatenate([inputs_train, inputs_test])
    labels = np.concatenate([labels_train, labels_test])
    inputs, labels = zip(*np.random.permutation(zip(inputs, labels)))

    model = LinearSVC()
    model.fit(inputs, labels)

    print 'Save Model'
    if save:
        save_model(model, scaler, path=save_path)

    return model, scaler