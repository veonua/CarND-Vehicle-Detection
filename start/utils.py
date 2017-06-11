import numpy as np
import cv2
import skvideo.io
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

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
