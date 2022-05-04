import os
import os.path as osp
import numpy as np
from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl, mano, flame
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, gen_trans_from_patch_cv
from utils.vis import vis_keypoints, vis_mesh, save_obj

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform='None', data_split='train'):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')

        # mscoco joint set
        self.joint_set = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Head', 'Pelvis', 'Neck', 'Torso'),
                            'flip_pairs': ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) , (18, 21), (19, 22), (20, 23), (24, 28), (25, 29) ,(26, 30), (27, 31) ),
                            'skeleton': ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
                            },\
                    'hand': \
                            {'joint_num': 21,
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            },
                    'face': \
                            {
                            'joint_to_flame': (-1, -1, -1, -1, -1, # no joints for neck, backheads, eyeballs
                                            17, 18, 19, 20, 21, # right eyebrow
                                            22, 23, 24, 25, 26, # left eyebrow
                                            27, 28, 29, 30, # nose
                                            31, 32, 33, 34, 35, # below nose
                                            36, 37, 38, 39, 40, 41, # right eye
                                            42, 43, 44, 45, 46, 47, # left eye
                                            48, # right lip
                                            49, 50, 51, 52, 53, # top lip
                                            54, # left lip
                                            55, 56, 57, 58, 59, # down lip
                                            60, 61, 62, 63, 64, 65, 66, 67, # inside of lip
                                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 # face contour
                                            )
                            }
                        }
        self.datalist, pose_list = self.load_data()
        self.pose_list = np.array(pose_list)
        print("COCO len: ", len(self.datalist))
    
    def add_joints(self, joint_coord):
        # pelvis
        lhip_idx = self.joint_set['body']['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['body']['joints_name'].index('R_Hip')
        pelvis = (joint_coord[lhip_idx,:] + joint_coord[rhip_idx,:]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
        pelvis = pelvis.reshape(1,3)
        
        # neck
        lshld_idx = self.joint_set['body']['joints_name'].index('L_Shoulder')
        rshld_idx = self.joint_set['body']['joints_name'].index('R_Shoulder')
        neck = (joint_coord[lshld_idx,:] + joint_coord[rshld_idx,:]) * 0.5
        neck[2] = joint_coord[lshld_idx,2] * joint_coord[rshld_idx,2]
        neck = neck.reshape(1,3)

        # torso
        torso = (neck + pelvis) * 0.5
        torso[:,2] = (torso[:,2] > 0).astype(np.float32)

        joint_coord = np.concatenate((joint_coord, pelvis, neck, torso)).astype(np.float32)
        return joint_coord

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            self.db_mask = COCO(osp.join(self.annot_path, 'instances_train2017.json'))
            # if cfg.parts == 'body':
            #     with open(osp.join(self.annot_path, 'MSCOCO_train_SMPL_NeuralAnnot.json')) as f:
            #         smpl_params = json.load(f)
            # if cfg.parts == 'hand':
            #     with open(osp.join(self.annot_path, 'MSCOCO_train_MANO_NeuralAnnot.json')) as f:
            #         mano_params = json.load(f)
            # if cfg.parts == 'face':
            #     with open(osp.join(self.annot_path, 'coco_smplifyx_train_FLAME.json')) as f:
            #         flame_params = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))


        pose_list = []
        # train mode
        if self.data_split == 'train':
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                # body part
                if cfg.parts == 'body':
                    if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                        continue
                    
                    # bbox
                    bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
                    if bbox is None: continue
                    
                    # joint coordinates
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                    # move head to last element
                    joint_img = np.concatenate([joint_img[1:], joint_img[:1]])

                    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                    if joint_valid.sum() < 10:
                        continue
                    joint_img = self.add_joints(joint_img)
                    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)

                    # if str(aid) in smpl_params:
                    #     smpl_param = smpl_params[str(aid)]
                    # else:
                    smpl_param = None

                    # transform -> 220x220
                    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
                    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
                    bb_width = float(bbox[2])
                    bb_height = float(bbox[3])
                    img2bb_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_img_shape[1], cfg.input_img_shape[0], 1.0, 0.0)

                    data_dict = {'img_path': img_path, 'img2bb_trans': img2bb_trans, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'smpl_param': smpl_param, 'ann_id': aid}
                    datalist.append(data_dict)

                    # affine transformation
                    joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, :1])), 1)
                    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)
                    # joint_valid = joint_valid * ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < cfg.input_img_shape[1]) * (joint_img[:, 1] >= 0) * (joint_img[:, 1] < cfg.input_img_shape[0])).reshape(-1, 1).astype(np.float32)
                    joint_img[:,2:] = joint_valid

                    pose_list.append(joint_img)

                    if len(datalist) > 40000:
                        break

            return datalist, pose_list

        # test mode
        else:
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                if cfg.parts == 'body':
                    bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
                    if bbox is None: continue

                    data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox}
                    datalist.append(data_dict)

                elif cfg.parts == 'hand':
                    for hand_type in ('right','left'):
                        if ann[hand_type + 'valid'] is False:
                            continue
                        bbox = ann[hand_type + 'box']
                        bbox = process_bbox(bbox, img['width'], img['height'])
                        if bbox is None:
                            continue
                        data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'hand_type': hand_type}
                        datalist.append(data_dict)

                elif cfg.parts == 'face':
                    if ann['valid'] is False:
                        continue
                    bbox = ann['box']
                    bbox = process_bbox(bbox, img['width'], img['height'])
                    if bbox is None:
                        continue

                    data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox}
                    datalist.append(data_dict)
            return datalist
 
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # train mode
        if self.data_split == 'train':
            img_path, img_shape = data['img_path'], data['img_shape']
            
            # image load
            img = load_img(img_path)
            
            # body part
            if cfg.parts == 'body':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.
                """ 
                # segmentation
                mask = self.db_mask.annToMask(self.db_mask.anns[data['ann_id']])
                mask = (mask > 0).astype(np.float32)
                if do_flip:
                    mask = mask[:, ::-1]
                mask = cv2.warpAffine(mask, img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR).astype(np.float32)
                mask = cv2.resize(mask, (cfg.input_img_shape[1]//8, cfg.input_img_shape[0]//8)) > 0
                mask = self.transform(mask.astype(np.float32))
                #_img = img * mask
                #cv2.imwrite(str(idx) + '.jpg', _img.numpy().transpose(1,2,0)*255)
                """

                # coco gt
                dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

                # smpl fitted data
                smpl_param = data['smpl_param']
                if smpl_param is not None:
                    smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param['smpl_param'], smpl_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'smpl')
                    smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
                    smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
                    smpl_shape_valid = float(True)

                    """
                    # for debug
                    _tmp = smpl_joint_img.copy() 
                    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                    _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                    _img = vis_keypoints(_img, _tmp)
                    cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                    """
                    
                else:
                    # dummy values
                    smpl_joint_img = np.zeros((smpl.joint_num,3), dtype=np.float32)
                    smpl_joint_cam = np.zeros((smpl.joint_num,3), dtype=np.float32)
                    smpl_joint_trunc = np.zeros((smpl.joint_num,1), dtype=np.float32)
                    smpl_pose = np.zeros((smpl.orig_joint_num*3), dtype=np.float32)
                    smpl_shape = np.zeros((smpl.shape_param_dim), dtype=np.float32)
                    smpl_joint_valid = np.zeros((smpl.joint_num,1), dtype=np.float32)
                    smpl_pose_valid = np.zeros((smpl.orig_joint_num*3), dtype=np.float32)
                    smpl_shape_valid = float(False)

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smpl_joint_img': smpl_joint_img, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(False)}#, 'mask': mask}
                return inputs, targets, meta_info

            # hand part
            elif cfg.parts == 'hand':
                # affine transform
                bbox, hand_type = data['bbox'], data['hand_type']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip=(hand_type=='left')) # enforce flip when left hand to make it right hand
                img = self.transform(img.astype(np.float32))/255.
            
                # coco gt
                dummy_coord = np.zeros((self.joint_set['hand']['joint_num'],3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans, rot, self.joint_set['hand']['joints_name'], mano.joints_name)

                # mano fitted data
                mano_param = data['mano_param']
                if mano_param is not None:
                    mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_mesh_cam_orig = process_human_model_output(mano_param['mano_param'], mano_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'mano')
                    mano_joint_valid = np.ones((mano.joint_num,1), dtype=np.float32)
                    mano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                    mano_shape_valid = float(True)

                    """
                    # for debug
                    _tmp = joint_img.copy()
                    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                    _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                    _img = vis_keypoints(_img, _tmp)
                    cv2.imwrite('coco_' + str(idx) + '_' + hand_type + '.jpg', _img)
                    _tmp = mano_joint_img.copy()
                    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                    _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                    _img = vis_keypoints(_img, _tmp)
                    cv2.imwrite('coco_' + str(idx) + '_' + hand_type + '_mano.jpg', _img)
                    """

                else:
                    # dummy values
                    mano_joint_img = np.zeros((mano.joint_num,3), dtype=np.float32)
                    mano_joint_cam = np.zeros((mano.joint_num,3), dtype=np.float32)
                    mano_joint_trunc = np.zeros((mano.joint_num,1), dtype=np.float32)
                    mano_pose = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                    mano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
                    mano_joint_valid = np.zeros((mano.joint_num,1), dtype=np.float32)
                    mano_pose_valid = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
                    mano_shape_valid = float(False)

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'is_3D': float(False)}
                return inputs, targets, meta_info

            # face part
            elif cfg.parts == 'face':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.

                # coco gt
                dummy_coord = np.zeros((flame.joint_num,3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, flame.flip_pairs, img2bb_trans, rot, flame.joints_name, flame.joints_name)

                # flame fitted data
                flame_param = data['flame_param']
                if flame_param is not None:
                    flame_joint_img, flame_joint_cam, flame_joint_trunc, flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, flame_joint_cam_orig, flame_mesh_cam_orig = process_human_model_output(flame_param['flame_param'], flame_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'flame')
                    flame_joint_valid = np.ones((flame.joint_num,1), dtype=np.float32)
                    flame_root_pose_valid = float(True)
                    flame_jaw_pose_valid = float(True)
                    flame_shape_valid = float(True)
                    flame_expr_valid = float(True)

                    """
                    # for debug
                    #_tmp = flame_joint_img.copy()
                    _tmp = joint_img.copy()
                    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                    _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                    _img = vis_keypoints(_img, _tmp)
                    cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                    """

                else:
                    # dummy values
                    flame_joint_cam = np.zeros((flame.joint_num,3), dtype=np.float32)
                    flame_root_pose = np.zeros((3), dtype=np.float32)
                    flame_jaw_pose = np.zeros((3), dtype=np.float32)
                    flame_shape = np.zeros((flame.shape_param_dim), dtype=np.float32)
                    flame_expr = np.zeros((flame.expr_code_dim), dtype=np.float32)
                    flame_joint_valid = np.zeros((flame.joint_num,1), dtype=np.float32)
                    flame_root_pose_valid = float(False)
                    flame_jaw_pose_valid = float(False)
                    flame_shape_valid = float(False)
                    flame_expr_valid = float(False)

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'flame_joint_cam': flame_joint_cam, 'flame_root_pose': flame_root_pose, 'flame_jaw_pose': flame_jaw_pose, 'flame_shape': flame_shape, 'flame_expr': flame_expr}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'flame_joint_valid': flame_joint_valid, 'flame_root_pose_valid': flame_root_pose_valid, 'flame_jaw_pose_valid': flame_jaw_pose_valid, 'flame_shape_valid': flame_shape_valid, 'flame_expr_valid': flame_expr_valid, 'is_3D': float(False)}
                return inputs, targets, meta_info

        # test mode
        else:
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)

            # body part
            if cfg.parts == 'body':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.
                
                inputs = {'img': img}
                targets = {}
                meta_info = {'bb2img_trans': bb2img_trans}
                return inputs, targets, meta_info

            # hand parts
            elif cfg.parts == 'hand':
                # affine transform
                bbox, hand_type = data['bbox'], data['hand_type']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split, enforce_flip=(hand_type=='left'))
                img = self.transform(img.astype(np.float32))/255.

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                return inputs, targets, meta_info

            # face parts
            elif cfg.parts == 'face':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                return inputs, targets, meta_info
    
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]

            if cfg.parts == 'body':
                vis = False
                if vis:
                    img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                    cv2.imwrite(str(ann_id) + '.jpg', img)
                    save_obj(out['smpl_mesh_cam'], smpl.face, str(ann_id) + '_body.obj')

                    # save SMPL parameter
                    bbox = annot['bbox']
                    smpl_pose = out['smpl_pose']; smpl_shape = out['smpl_shape']; smpl_trans = out['cam_trans']
                    focal_x = cfg.focal[0] / cfg.input_img_shape[1] * bbox[2]
                    focal_y = cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]
                    princpt_x = cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0]
                    princpt_y = cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]
                    save_dict = {'smpl_param': {'pose': smpl_pose.reshape(-1).tolist(), 'shape': smpl_shape.reshape(-1).tolist(), 'trans': smpl_trans.reshape(-1).tolist()},\
                                'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                                }
                    with open(osp.join(save_path, 'smpl_param.json'), 'w') as f:
                        json.dump(save_dict, f)

            elif cfg.parts == 'hand':
                vis = False
                if vis:
                    save_obj(out['mano_mesh_cam'], mano.face['right'], str(ann_id) + '_rhand.obj')
                    save_obj(out['mano_mesh_cam'], mano.face['left'], str(ann_id) + '_lhand.obj')

                    # all hands are flipped to the right hand.
                    # restore to the left hand.
                    if hand_type == 'left':
                        mano_pose = mano_pose.reshape(-1,3)
                        mano_pose[:,1:3] *= -1
                        mano_trans[0] *= -1

                    focal_x = cfg.focal[0] / cfg.input_img_shape[1] * bbox[2]
                    focal_y = cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]
                    princpt_x = cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0]
                    princpt_y = cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]
                    save_dict = {'mano_param': {'pose': mano_pose.reshape(-1).tolist(), 'shape': mano_shape.reshape(-1).tolist(), 'trans': mano_trans.reshape(-1).tolist()},\
                                'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                                }
                    with open(osp.join(save_path, 'mano_param_' + hand_type + '.json'), 'w') as f:
                        json.dump(save_dict, f)

            
            elif cfg.parts == 'face':
                vis = False
                if vis:
                    img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                    cv2.imwrite(str(ann_id) + '.jpg', img)
                    save_obj(out['flame_mesh_cam'], flame.face, str(ann_id) + '_face.obj')

                    focal_x = cfg.focal[0] / cfg.input_img_shape[1] * bbox[2]
                    focal_y = cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]
                    princpt_x = cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0]
                    princpt_y = cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]
                    save_dict = {'flame_param': {'root_pose': flame_root_pose.reshape(-1).tolist(), 'jaw_pose': flame_jaw_pose.reshape(-1).tolist(), 'shape': flame_shape.reshape(-1).tolist(), 'expr': flame_expr.reshape(-1).tolist(), 'trans': flame_trans.reshape(-1).tolist()},\
                                'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                                }
                    with open(osp.join(save_path, 'flame_param.json'), 'w') as f:
                        json.dump(save_dict, f)


        return {}

    def print_eval_result(self, eval_result):
        return
