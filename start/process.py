import numpy as np

import patches
import pipeline as p
import cv2

import _init_paths
from model.test import im_detect
from model.nms_wrapper import nms

#from math import inf

_debug = False
_ploty = None
_ool, _oor = None, None
_sess, _net = None, None


def process_image(original):
    global _ploty, _ool, _oor, lleft_x, lleft_y, lright_x, lright_y, _sess, _net

    dst = p.undistort(original)

    # Farster RCNN
    scores, boxes = im_detect(_sess, _net, dst)
    
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    cls_ind = 7 # car

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    
    if len(inds) > 0:
        for i in inds:
            bbox = dets[i, :4].astype(int)
            score = dets[i, -1]
            cv2.rectangle(dst,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,(int)(score*255),0),3)
    #

    warped, sure = p.get_birdview(dst)
    edges = p.get_lines(warped)
    sm = np.array(edges[-edges.shape[0] // 4:, :])

    if _ploty is None:
        _ploty = np.linspace(0, sm.shape[0] - 1, num=sm.shape[0])

    left_x, left_y, right_x, right_y = patches.find_curves2(sm, _ool, _oor, de2=None, verbose=5 if _debug else 0)
    left_fitx, right_fitx, _ool, _oor = patches.polyfit(_ploty, left_x, left_y, right_x, right_y)

    if (_ool is not None) and (_oor is not None) and ((_oor - _ool > 16) or (_oor - _ool < 6)):
        _ool, _oor = None, None

    valid = False
    if left_fitx is not None and right_fitx is not None:
        left_curvature, right_curvature, calculated_deviation = p.calculate_info(left_x, left_y, right_x, right_y)
        if abs(calculated_deviation) < 0.6:
            valid = True
            lleft_x, lleft_y, lright_x, lright_y = left_x, left_y, right_x, right_y

    if not valid:
        left_x, left_y, right_x, right_y = lleft_x, lleft_y, lright_x, lright_y
        left_fitx, right_fitx, _ool, _oor = patches.polyfit(_ploty, left_x, left_y, right_x, right_y)
        left_curvature, right_curvature, calculated_deviation = p.calculate_info(left_x, left_y, right_x, right_y)

    layer = p.fill_lane_lines(np.dstack((edges, edges, edges)).astype('uint8'), np.add(_ploty, 300), left_fitx,
                              right_fitx)

    res = p.merge_marks(dst, layer)

    #
    if left_curvature > 99999:
        left_curvature = float("inf")

    if right_curvature > 99999:
        right_curvature = float("inf")

    curvature_text = 'Curvature left: {:.2f} m    right: {:.2f} m'.format(left_curvature, right_curvature)
    cv2.putText(res, curvature_text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (28, 221, 119), 2)

    deviation_info = 'Lane Deviation: {:.2f} m'.format(calculated_deviation)
    cv2.putText(res, deviation_info, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (28, 221, 119), 2)
    return res
