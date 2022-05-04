import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.layer import make_conv_layers, make_linear_layers
from utils.human_models import smpl, mano, flame
from utils.transforms import sample_image_feature, soft_argmax_2d, soft_argmax_3d
from config import cfg

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        self.conv = make_conv_layers([2048,self.joint_num*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1,self.joint_num,cfg.output_hm_shape[0]*cfg.output_hm_shape[1]*cfg.output_hm_shape[2]),2)
        joint_hm = joint_hm.view(-1,self.joint_num,cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        return joint_hm, joint_coord

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
       
        # output layers
        if cfg.parts == 'body':
            self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(512+3), (smpl.orig_joint_num-3)*6], relu_final=False) # without root and two hands
            self.shape_out = make_linear_layers([2048,smpl.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([2048,3], relu_final=False)
        elif cfg.parts == 'hand':
            self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(512+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
            self.shape_out = make_linear_layers([2048,mano.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([2048,3], relu_final=False)

    def sample_image_feature_joint(self, img_feat, joint_xy):
        joint_num = joint_xy.shape[1]
        img_feat_joints = []
        for j in range(joint_num):
            img_feat_joints.append(sample_image_feature(img_feat, joint_xy[:,j,:]))
        img_feat_joints = torch.stack(img_feat_joints,1)
        return img_feat_joints

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        img_feat = self.conv(img_feat)
        img_feat_joints = self.sample_image_feature_joint(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)
        if cfg.parts == 'body':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        elif cfg.parts == 'hand':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.root_pose_out = make_linear_layers([512,6], relu_final=False) # root pose
        self.jaw_pose_out = make_linear_layers([512,6], relu_final=False) # jaw pose
        self.shape_out = make_linear_layers([512, flame.shape_param_dim], relu_final=False) # shape parameter
        self.expr_out = make_linear_layers([512, flame.expr_code_dim], relu_final=False) # expression parameter
        self.cam_out = make_linear_layers([512,3], relu_final=False) # camera parameter

    def forward(self, img_feat):
        feat = img_feat.mean((2,3))
        
        # pose parameter
        root_pose = self.root_pose_out(feat)
        jaw_pose = self.jaw_pose_out(feat)

        # shape parameter
        shape_param = self.shape_out(feat)

        # expression parameter
        expr_param = self.expr_out(feat)

        # camera parameter
        cam_param = self.cam_out(feat)

        return root_pose, jaw_pose, shape_param, expr_param, cam_param

