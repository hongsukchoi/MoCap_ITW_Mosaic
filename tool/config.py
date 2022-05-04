import os
import os.path as osp
import sys
import numpy as np


class Config:
    ## dataset
    trainset_3d = ['Human36M']
    trainset_2d = []
    testset = 'Human36M'

    ## model setting
    resnet_type = 50

    ## training config
    lr = 1e-4
    lr_dec_factor = 10
    lr_dec_epoch = [40, 60]  # [4,6], [10,12]
    end_epoch = 70  # 7, 13
    train_batch_size = 48
    ## input, output
    input_img_shape = (256, 192)  # (256, 192)
    output_hm_shape = (64, 64, 64)  # (64, 64, 48)

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 40  # 16
    gpu_ids = '0'
    num_gpus = 1
    parts = 'body'
    continue_train = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, gpu_ids, parts, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.parts = parts
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

        if self.parts == 'body':
            self.bbox_3d_size = 2
            self.camera_3d_size = 2.5
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
        elif self.parts == 'hand':
            self.bbox_3d_size = 0.3
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 256)
            self.output_hm_shape = (8, 8, 8)
        elif self.parts == 'face':
            self.bbox_3d_size = 0.3
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
        else:
            assert 0, 'Unknown parts: ' + self.parts

        self.focal = (5000, 5000)  # virtual focal lengths
        self.princpt = (self.input_img_shape[1] / 2, self.input_img_shape[0] / 2)  # virtual principal point position


cfg = Config()

