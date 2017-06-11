import matplotlib.image as mpimg
import process
from moviepy.editor import VideoFileClip
import os
import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

dataset = "voc_2007_trainval+voc_2012_trainval"
nnet    = "res101_faster_rcnn_iter_110000.ckpt"

tfmodel = os.path.join("/home/veon/edu/tf-faster-rcnn-master/data", dataset, nnet)
#'../data', dataset, nnet)


if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth=True

# init session
sess = tf.Session(config=tfconfig)
 

if nnet.startswith('vgg16'):
    net = vgg16(batch_size=1)
elif nnet.startswith('res101'):
    net = resnetv1(batch_size=1, num_layers=101)
    

net.create_architecture(sess, "TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
saver = tf.train.Saver()
saver.restore(sess, tfmodel)

print('Loaded network {:s}'.format(tfmodel))

process._sess = sess
process._net = net

#process._debug = False

    
clip1 = VideoFileClip("../project_video.mp4")
video_clip = clip1.fl_image(process.process_image)
video_clip.write_videofile("../project_out1.mp4", codec='mpeg4', audio=False)
