import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

mtx = np.array([[1156.60, 0.00, 669.05], [0.00, 1151.67, 388.15], [0.00, 0.00, 1.00]])
dist = np.array([-0.23, -0.12, -0.00, 0.00, 0.16])

wR, hR = (400, 400)
confidence = np.expand_dims(np.transpose([np.append(np.zeros([hR // 2]), np.linspace(0, 1, num=hR // 2)), ] * wR), 2)
h, w = (720, 1200)

# get M, the transform matrix
M = None
Minv = None


def get_persp_points(h):
    p0 = (616, 380)  # 214* 2
    p1 = (p0[0] - 122, p0[1])
    p2 = (p0[0] + 122, p0[1])

    dd = 2400
    return np.float32([p1, p2, (w + dd, h), (-dd, h)])


def undistort(img):
    global h
    global w

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h), centerPrincipalPoint=False)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    r = dst[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    h, w = r.shape[:2]

    srcP = get_persp_points(h)
    dstP = np.float32([(0, 0), (wR, 0), (wR, hR), (0, hR)])

    global M
    global Minv
    # get M, the transform matrix
    M = cv2.getPerspectiveTransform(srcP, dstP)
    Minv = inv(M)
    return r


def get_birdview(img, save=None):
    # Perspective transform
    warped = cv2.warpPerspective(img, M, (wR, hR), flags=cv2.INTER_CUBIC)
    sure = warped[-hR // 4:-4, :, :]

    if save is not None:
        mpimg.imsave(save, warped)

    return warped, sure


def genereate_straight_lanes(img=None):
    if img is None:
        d = np.zeros((wR, hR, 3), np.uint8)
    else:
        d = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    l = int(266 / 600 * wR)
    ld = int(78 / 600 * hR)
    cv2.line(d, (l - ld, 0), (l - ld, hR), color=[0, 0, 255], thickness=2)
    cv2.line(d, (l, 0), (l, hR), color=[0, 0, 255], thickness=2)
    cv2.line(d, (l + ld, 0), (l + ld, hR), color=[0, 0, 255], thickness=2)
    cv2.line(d, (l + ld * 2, 0), (l + ld * 2, hR), color=[0, 0, 255], thickness=2)
    cv2.line(d, (l + ld * 3, 0), (l + ld * 3, hR), color=[0, 0, 255], thickness=2)
    return d


# debug
def draw_persp(dst):
    d = np.copy(dst)

    src_p = get_persp_points(dst.shape[0])

    cv2.line(d, tuple(src_p[2]), tuple(src_p[1]), color=[0, 255, 0], thickness=2)
    cv2.line(d, tuple(src_p[3]), tuple(src_p[0]), color=[0, 255, 0], thickness=2)

    p0 = (620, 380);
    cv2.line(d, p0, (w // 2, h), color=[0, 0, 170], thickness=3)

    plt.imshow(d)
    plt.show()


def merge_marks(img, marks):
    if len(marks.shape) == 2:
        r = (np.ones_like(marks) * 150)
        m = np.dstack((np.minimum(r, marks), marks, np.zeros_like(marks)))
    else:
        m = marks

    d = cv2.warpPerspective(m, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 0.5, d, 1, 0.0)


def draw_marks(img, marks):
    plt.imshow(merge_marks(img, marks))
    plt.show()


def get_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ll = np.array([[0, -1, 1, 0, 0]])
    rl = np.array([[0, 0, 0, 1, -1]])
    re1 = cv2.filter2D(hsv[:, :, 2], cv2.CV_32F, ll).clip(0)
    re3 = cv2.filter2D(hsv[:, :, 2], cv2.CV_32F, rl).clip(0)

    return np.divide(np.multiply(re1, re3), 25).astype(img.dtype)  # np.minimum(re1, re3).astype(img.dtype)


def fill_lane_lines(image, fit_y, fit_left_x, fit_right_x):
    copy_image = np.zeros_like(image)

    pts_left = np.array([np.transpose(np.vstack([fit_left_x, fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_right_x, fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(copy_image, np.int_([pts]), (0, 0, 160))

    return cv2.addWeighted(copy_image, 0.5, image, 1, 0)


def calculate_info(leftx, lefty, rightx, righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048 / 16  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 56    # meters per pixel in x dimension

    y_eval = 100

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(np.multiply(lefty, ym_per_pix), np.multiply(leftx, xm_per_pix), 2)
    right_fit_cr = np.polyfit(np.multiply(righty, ym_per_pix), np.multiply(rightx, xm_per_pix), 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Next, we can lane deviation
    base_x_left_m = left_fit_cr[0] * (y_eval * ym_per_pix) ** 2 \
                    + left_fit_cr[1] * (y_eval * ym_per_pix) + left_fit_cr[2]
    base_x_right_m = right_fit_cr[0] * (y_eval * ym_per_pix) ** 2 \
                     + right_fit_cr[1] * (y_eval * ym_per_pix) + right_fit_cr[2]
    line_center = (base_x_right_m + base_x_left_m) / 2
    image_center = xm_per_pix * wR / 2
    offset_m = line_center - image_center

    return left_curverad, right_curverad, offset_m
