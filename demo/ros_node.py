#!/usr/bin/python

import os, sys
from pathlib import Path
SCRIPT_PATH = Path(os.path.abspath(__file__))
sys.path.append(str(SCRIPT_PATH / '..'))

import numpy as np

import rospy
import torch
from sensor_msgs.msg import CompressedImage
import cv_bridge
import cv2
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.models.detectors.condlanenet import CondLanePostProcessor

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def adjust_point(hm_points,
                 downscale,
                 crop_bbox,
                 img_shape,
                 tgt_shape=(590, 1640)):
    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top
    coord_x = float((hm_points[0] + 0.5) * downscale * ratio_x + offset_x)
    coord_y = float((hm_points[1] + 0.5) * downscale * ratio_y + offset_y)
    coord_x = max(0, coord_x)
    coord_x = min(coord_x, tgt_shape[1])
    coord_y = max(0, coord_y)
    coord_y = min(coord_y, tgt_shape[0])
    return [coord_x, coord_y]


def out_result(lanes, dst=None):
    if dst is not None:
        with open(dst, 'w') as f:
            for lane in lanes:
                for idx, p in enumerate(lane):
                    if idx == len(lane) - 1:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f}'.format(p[1]), file=f)
                    else:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f} '.format(p[1]), end='', file=f)


class Processor:
    def __init__(self):
        self._cv_bride = cv_bridge.CvBridge()
        model = dict(
                type='CondLaneNet',
                pretrained='torchvision://resnet101',
                train_cfg=dict(out_scale=4),
                test_cfg=dict(out_scale=4),
                num_classes=1,
                backbone=dict(
                    type='ResNet',
                    depth=101,
                    strides=(1, 2, 2, 2),
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch'),
                neck=dict(
                    type='TransConvFPN',
                    in_channels=[256, 512, 1024, 256],
                    out_channels=64,
                    num_outs=4,
                    trans_idx=-1,
                    trans_cfg=dict(
                        in_dim=2048,
                        attn_in_dims=[2048, 256],
                        attn_out_dims=[256, 256],
                        strides=[1, 1],
                        ratios=[4, 4],
                        pos_shape=(1, 10, 25),
                    ),
                ),
                head=dict(
                    type='CondLaneHead',
                    heads=dict(hm=1),
                    in_channels=(64, ),
                    num_classes=1,
                    head_channels=64,
                    head_layers=1,
                    disable_coords=False,
                    branch_in_channels=64,
                    branch_channels=64,
                    branch_out_channels=64,
                    reg_branch_channels=64,
                    branch_num_conv=1,
                    hm_idx=2,
                    mask_idx=0,
                    compute_locations_pre=True,
                    location_configs=dict(size=(1, 1, 80, 200), device='cuda:0')),
                loss_weights=dict(
                    hm_weight=1,
                    kps_weight=0.4,
                    row_weight=1.,
                    range_weight=1.,
                ))
        model['pretrained'] = None
        model = build_detector(model)
        load_checkpoint(model, 'culane_large.pth', map_location='cpu')
        self.model = model.cuda().eval()

        self.post_processor = CondLanePostProcessor(mask_size=(1, 80, 200), hm_thr=0.3, seg_thr=0.3)
        self._subs = {
            'cam_fc_far': rospy.Subscriber('/cam_fc_far/image_rect_color/compressed',
                                           CompressedImage, self.image_callback)
        }

        self.colors = []
        for _ in range(20):
            color = np.random.randint(0, 255, (3,))
            color = [int(c) for c in color]
            self.colors.append(color)

    def image_callback(self, compessed_image: CompressedImage):
        image = self._cv_bride.compressed_imgmsg_to_cv2(compessed_image, "bgr8")
        image = image[300:, :, :]
        image = cv2.resize(image, (800, 320))
        mean = np.array([75.3, 76.6, 77.6])
        std = np.array([50.5, 53.8, 54.3])
        in_image = mmcv.imnormalize(image, mean, std, False)

        x = torch.unsqueeze(torch.from_numpy(in_image).permute(2, 0, 1), 0)
        x = x.cuda()
        seeds, _ = self.model.test_inference(x)
        lanes, seeds = self.post_processor(seeds, 4)

        for index, lane in enumerate(lanes):
            points = lane['points']
            for pt1, pt2, in zip(points[:-1], points[1:]):
                cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), self.colors[index], 4)

        cv2.imshow('lanes', image)
        cv2.waitKey(11)


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    processor = Processor()
    rospy.spin()
