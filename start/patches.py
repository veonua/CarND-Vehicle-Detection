import cv2
import numpy as np


def rebin(a, shape):
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).sum(-1).sum(1)


def max_pool(a, f):
    sh = a.shape[0] // f, f, a.shape[1] // f, f
    return a.reshape(sh).sum(-1).sum(1)


def get_max_ind(img, level, old_l, old_r, w_l=2, w_r=2):
    try:
        dy = 1
        zy = img.shape[0] - dy * (level + 1)
        l1, l2 = int(old_l - w_l), int(old_l + w_l)
        r1, r2 = int(old_r - w_r), int(old_r + w_r)

        left = img[zy:zy + dy, l1:l2].reshape(-1)
        right = img[zy:zy + dy, r1:r2].reshape(-1)

        li, ri = np.argmax(left), np.argmax(right)

        r = img[zy:zy + dy, l1 + li + 1:r1 + ri - 1].reshape(-1)
        # print (":::", np.mean(r), np.max(r))

        return l1 + li, r1 + ri, left[li], right[ri]

    except:
        print(level, old_l, old_r, w_l, w_r)
        print(l1, l2, r1, r2)


def _draw_rect2(de2, xx, yy, c, delta):
    win_xleft_low = int((xx - delta) * 4)
    win_y_low = int(yy * 4)
    win_xleft_high = int((xx + delta) * 4)
    win_y_high = int(yy * 4)
    cv2.rectangle(de2, (win_xleft_low, win_y_low),
                  (win_xleft_high, win_y_high),
                  c, 1)


def _draw_rect(de2, xx, yy, c):
    win_xleft_low = int(xx * 4 - 2)
    win_y_low = int(yy * 4 - 2)
    win_xleft_high = int(xx * 4 + 2)
    win_y_high = int(yy * 4 + 2)
    cv2.rectangle(de2, (win_xleft_low, win_y_low),
                  (win_xleft_high, win_y_high),
                  c, 1)


def _expand_window(window):
    max_window = 4
    if window >= max_window:
        return max_window
    return window + 0.33


def less_than_threshold(value, level):
    th = max(100 - level * 10, 35)
    return value < th


def find_curves2(edges_img, old_l=None, old_r=None, verbose=0, de2=None):
    downscaled = rebin(edges_img, (edges_img.shape[0] // 4, edges_img.shape[1] // 4))

    min_w = 99
    m = downscaled.shape[1] // 2

    if old_l is None:
        window_l = 4
    else:
        window_l = 3

    if old_r is None:
        window_r = 4
    else:
        window_r = 3

    lefty, leftx = [], []
    righty, rightx = [], []

    h = downscaled.shape[0]

    # old_l, old_r, = None, None, None  # m - delta_l, m + delta_r, None
    dl, dr = None, None

    for level in range(0, h):
        yy = h - level - 1

        if dl is None: dl = 0
        if dr is None: dr = 0

        if old_l is None: old_l = m - window_l - 2
        if old_r is None: old_r = m + window_r + 2
        # + dr // 2
        l, r, lv, rv = get_max_ind(downscaled, level, old_l, old_r, window_l, window_r)

        if de2 is not None:
            _draw_rect2(de2, old_l, yy, (0, 200, 0), window_l)
            _draw_rect2(de2, old_r, yy, (0, 0, 200), window_r)

        try:
            dl = l - old_l
        except:
            dl = None

        try:
            dr = r - old_r
        except:
            dr = None

        if less_than_threshold(lv, level):  # old_m - l < 2 or (np.abs(dl) > 3) and np.sign(dl) != np.sign(old_dm)):
            dl = None
            l = None
            window_l = _expand_window(window_l)

        else:
            lefty.append((h - level) * 4)
            leftx.append(l * 4)
            window_l = 2
            if de2 is not None:
                _draw_rect(de2, l, yy, (0, lv * 5, 0))

        if less_than_threshold(rv, level):  # r - old_m or (np.abs(dr) > 3) and np.sign(dr) != np.sign(old_dm)):
            dr = None
            r = None
            window_r = _expand_window(window_r)
        else:
            righty.append((h - level) * 4)
            rightx.append(r * 4)
            window_r = 2
            if de2 is not None:
                _draw_rect(de2, r, yy, (0, 0, lv * 5))

        old_width = old_r - old_l

        case = ""

        if r is None and l is None:
            l = old_l
            r = old_r
            m = old_l + old_width / 2
            case = "*"
        elif r is None:
            r = l + old_width
            m = l + old_width / 2
            case = ">"
        elif l is None:
            l = r - old_width
            m = l + old_width / 2
            case = "<"
        else:
            w = r - l
            min_w = min((w, min_w))
            m = l + w / 2

        if verbose > 3:
            try:
                w = r - l
            except:
                w = -1

            print(h - level, ":", l, m, r, ";", lv, rv, w, case)

        (old_l, old_r) = (l, r)

    return leftx, lefty, rightx, righty


# def find_curves(edges_img, ploty, old_l=None, old_r=None, verbose=0, de2=None):
#     leftx, lefty, rightx, righty = find_curves2(edges_img, old_l, old_r, verbose=verbose, de2=de2)
#
#     return polyfit(ploty, leftx, lefty, rightx, righty, verbose)

def polyfit(ploty, left_x, left_y, right_x, right_y, verbose=0):
    if len(left_x) > 2:
        left_fit = np.polyfit(left_y, left_x, 2)
        if verbose > 2:
            print(['{:.3f}'.format(i) for i in left_fit])
        left_fitx = np.add(left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2], 2)
        ool = left_fitx[-10] // 4
    else:
        left_fitx = None
        ool = None

    if len(right_x) > 2:
        right_fit = np.polyfit(right_y, right_x, 2)
        if verbose > 2:
            print(['{:.3f}'.format(i) for i in right_fit])
        right_fitx = np.add(right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2], 2)
        oor = right_fitx[-10] // 4
    else:
        right_fitx = None
        oor = None

    return left_fitx, right_fitx, ool, oor