from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
        self.pretrian_model_path = './pretrain_model/pose_hrnet_w32_256x256.pth'
        # self.pretrian_model_path = './snapshot/foot_robot1028/foot_robot1028_512_512_160.pth'
        #训练参数及超参数
        self.net_width   = 512
        self.net_height  = 512
        self.max_epochs  = 3000
        self.lr          = 0.0001
        self.dataload_num_workers = 16
        self.device_ids = [0,1,2,3]
        self.batch_size = 1 * len(self.device_ids)
        # 学习率策略
        self.lr_scheduer = {
             5: self.lr * len(self.device_ids),
            # 80: self.lr * 0.1,
            # 120:self.lr *0.01
        }
        #
        self.task_name = "foot_robot1206"
        self.mean    = [0.408, 0.447, 0.47]
        self.std     = [0.289, 0.274, 0.278]
        
        self.train_dir = "/home/fsw/Documents/codes/CenterNet_match/dataset/foot_robot/train"
        self.class_names = ['label1', 'label2']
         
        self.label_map = dict(zip(self.class_names, [i for i in range(len(self.class_names))]))
      
      #   self.label_map = {
      #         'obj': 0
      #   }  # 单一类别

        self.num_classes = len(self.label_map)


        #data模块参数
        self.not_rand_crop = False
        self.flip = 0.5
        self.down_ratio = 4
        self.reg_offset = True

        #loss模块参数
        self.num_stacks = 1
        self.em_weight = 1.
        self.wh_weight = 0.4
        self.hm_weight = 1.
        self.off_weight = 1.
        self.mse_loss = False
        self.reg_loss = 'l1'
        self.dense_wh    = False  #  直接回归长宽度的map
        self.norm_wh     = False
        self.cat_spec_wh = False

        self.max_objs = 128

        self.need_sigmoid = False
