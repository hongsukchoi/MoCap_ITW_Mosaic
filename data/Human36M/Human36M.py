import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.human_models import smpl
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, gen_trans_from_patch_cv
from utils.transforms import world2cam, cam2pixel, rigid_align
from utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
from glob import glob
import random


class Human36M(torch.utils.data.Dataset):
    def __init__(self, subject, part):
        self.subject = subject
        self.part = part
        self.num_parts = 10

        self.transform = None
        self.data_split = 'train'
        self.img_dir = osp.join('..', 'data', 'Human36M', 'images')
        self.annot_path = osp.join('..', 'data', 'Human36M', 'annotations')
        self.itw_img_dir = osp.join('..', 'data', 'MSCOCO', 'images')
        self.itw_annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        # H36M joint set
        self.joint_set = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
                            'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
                            # 'neighbor_joints': {0: [7,1,4], 1: [0,2], 2: [1,3], 3: [2,], 4: [0,5], 5: [4,6], 6: [5], 7: [8,], 8: [7,9,11], 9: [8,10], 10: [9,], 11: [8,12], 12: [11,13], 13: [12,], 14: [8,15], 15: [14,16], 16: [15] },
                            # 'neighbor_joints': {0: [7, 1, 4], 1: [2], 2: [1, 3], 3: [2, ], 4: [ 5], 5: [4, 6], 6: [5], 7: [8, ], 8: [7, 9, 11], 9: [8, 10], 10: [9, ], 11: [12], 12: [11, 13], 13: [12, ],
                            #                      14: [ 15], 15: [14, 16], 16: [15]},
                            'neighbor_joints': {0: [7,1,4], 1: [0,2], 2: [1,3], 3: [2,], 4: [0,5], 5: [4,6], 6: [5], 7: [8,], 8: [7,9,11], 9: [8], 10: [9,], 11: [8,12], 12: [11,13], 13: [12,], 14: [8,15], 15: [14,16], 16: [15] },

                             'skeleton': ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) ),
                            'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                            'smpl_regressor': np.load(osp.join('..', 'data', 'Human36M', 'J_regressor_h36m_smpl.npy')),
                            }
                        }
        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()
        print("Human36M len: ", len(self.datalist))
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [self.subject]  # [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            # with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPL_NeuralAnnot.json'),'r') as f:
            #     smpl_params[str(subject)] = json.load(f)

        db.createIndex()

        # # itw dataset
        # self.itw_ann_list = []
        # self.db_itw = COCO(osp.join(self.itw_annot_path, 'instances_train2017.json'))
        # for aid in self.db_itw.anns.keys():
        #     ann = self.db_itw.anns[aid]
        #     img = self.db_itw.loadImgs(ann['image_id'])[0]
        #     imgname = osp.join('train2017', img['file_name'])
        #     img_path = osp.join(self.itw_img_dir, imgname)
        #
        #     if ann['iscrowd'] or ann['category_id'] != 1:
        #         continue
        #
        #     # bbox
        #     if ann['bbox'][2] * ann['bbox'][3] < 128*128:
        #         continue
        #     #if img['height'] < cfg.input_img_shape[0] or img['width'] < cfg.input_img_shape[1]:
        #     #    continue
        #     bbox = process_bbox(ann['bbox'], img['width'], img['height'])
        #     if bbox is None: continue
        #
        #     data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'ann_id': aid}
        #     self.itw_ann_list.append(data_dict

        datalist = []
        total_length = len(db.anns.keys())
        partition = total_length // self.num_parts
        start = partition * self.part
        end = partition * (self.part + 1) if self.part < (self.num_parts -1) else total_length
        keys = sorted(list(db.anns.keys()))[start:end]

        for aid in keys:
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            # try:
            #     smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            # except KeyError:
            smpl_param = None

            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)[:,:2]
            joint_valid = np.ones((self.joint_set['body']['joint_num'],1))

            ref_joint_idx = []
            for j in range(len(joint_img)):
                nei_joints = self.joint_set['body']['neighbor_joints'][j]
                max_idx = nei_joints[0]
                max_dist = np.sqrt(np.sum((joint_img[j] - joint_img[max_idx])**2))
                for n in nei_joints:
                    dist = np.sqrt(np.sum((joint_img[j] - joint_img[n])**2))
                    if dist > max_dist:
                        max_idx = n
                        max_dist = dist
                ref_joint_idx.append(self.joint_set['body']['joints_name'][max_idx])
            ref_joint_idx = np.array(ref_joint_idx)

            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue

            # transform -> 220x220
            bb_c_x = float(bbox[0] + 0.5 * bbox[2])
            bb_c_y = float(bbox[1] + 0.5 * bbox[3])
            bb_width = float(bbox[2])
            bb_height = float(bbox[3])

            img2bb_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_img_shape[1], cfg.input_img_shape[0], 1.0, 0.0)

            # affine transformation
            joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, :1])), 1)
            joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)

            datalist.append({
                # 'inv_img2bb_trans': inv_img2bb_trans,
                'img2bb_trans': img2bb_trans,
                'img_name': img['file_name'],
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'ref_joint_name': ref_joint_idx,
                'smpl_param': smpl_param,
                'cam_param': cam_param})

            # if len(datalist) > 10:
            #     break

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        if self.data_split == 'train':
            # h36m gt
            joint_cam = data['joint_cam']
            joint_cam = (joint_cam - joint_cam[self.joint_set['body']['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)
            
            smpl_param = data['smpl_param']
            if smpl_param is not None:
                # smpl coordinates and parameters
                cam_param['t'] /= 1000 # milimeter to meter
                smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')
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
                cv2.imwrite('h36m_' + str(idx) + '.jpg', _img)
                """
                
                # mask
                smpl_mesh_img = smpl_mesh_cam_orig[:,:2] / smpl_mesh_cam_orig[:,2:] * cam_param['focal'].reshape(1,2) + cam_param['princpt'].reshape(1,2)
                mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
                x = smpl_mesh_img[:,0]
                y = smpl_mesh_img[:,1]
                x_center = (x[smpl.face[:,0]] + x[smpl.face[:,1]] + x[smpl.face[:,2]])/3.
                y_center = (y[smpl.face[:,0]] + y[smpl.face[:,1]] + y[smpl.face[:,2]])/3.
                x = np.concatenate((x, x_center, (x_center + x[smpl.face[:,0]])/2., (x_center + x[smpl.face[:,1]])/2., (x_center + x[smpl.face[:,2]])/2.))
                y = np.concatenate((y, y_center, (y_center + y[smpl.face[:,0]])/2., (y_center + y[smpl.face[:,1]])/2., (y_center + y[smpl.face[:,2]])/2.))
                x[x < 0] = 0; x[x > img_shape[1] - 1] = img_shape[1] - 1
                y[y < 0] = 0; y[y > img_shape[0] - 1] = img_shape[0] - 1
                mask[y.astype(np.int32),x.astype(np.int32)] = 1.0
                mask = cv2.GaussianBlur(mask, (3,3), 0)
                mask = (mask > 0.15).astype(np.float32)
                if do_flip:
                    mask = mask[:, ::-1]
                mask = cv2.warpAffine(mask, img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR).astype(np.float32) > 0
                # for debug
                #_mask = self.transform(mask.astype(np.float32))
                #_img = img * _mask
                #cv2.imwrite(str(idx) + '.jpg', _img.numpy().transpose(1,2,0)[:,:,::-1]*255)
                #
                #mask = cv2.resize(mask.astype(np.uint8), (cfg.input_img_shape[1]//8, cfg.input_img_shape[0]//8)) > 0
                mask = self.transform(mask.astype(np.float32))
                
                """
                # bkg copy and paste
                fg = img * mask; bg = img * (1 - mask);
                itw_data = self.itw_ann_list[random.randint(0, len(self.itw_ann_list)-1)]
                itw_img = load_img(itw_data['img_path'])
                xmin = random.randint(0,itw_img.shape[1] - cfg.input_img_shape[1])
                ymin = random.randint(0,itw_img.shape[0] - cfg.input_img_shape[0])
                itw_img = itw_img[ymin:ymin+cfg.input_img_shape[0], xmin:xmin+cfg.input_img_shape[1], :]
                itw_img = self.transform(itw_img.astype(np.float32))/255.
                img = img * mask + itw_img * (1 - mask)
                #cv2.imwrite(str(idx) + '.jpg', img.numpy().transpose(1,2,0)[:,:,::-1]*255)
                """
                
                # style mix
                fg = img * mask
                mean_fg = torch.sum(fg,(1,2))[:,None,None] / (torch.sum(mask,(1,2))[:,None,None] + 1e-5)
                std_fg = torch.sqrt((torch.sum((fg + (1 - mask) * mean_fg - mean_fg)**2, (1,2)) + 1e-5) / (torch.sum(mask,(1,2)) + 1e-5))[:,None,None]

                itw_data = self.itw_ann_list[random.randint(0, len(self.itw_ann_list)-1)]
                itw_img_path, itw_img_shape, itw_bbox, itw_mask = itw_data['img_path'], itw_data['img_shape'], itw_data['bbox'], self.db_itw.annToMask(self.db_itw.anns[itw_data['ann_id']]) 
                itw_img = load_img(itw_img_path)
                itw_img, itw_img2bb_trans, _, itw_rot, itw_do_flip = augmentation(itw_img, itw_bbox, self.data_split)
                itw_img = self.transform(itw_img.astype(np.float32))/255.
                itw_mask = (itw_mask > 0).astype(np.float32)
                if itw_do_flip:
                    itw_mask = itw_mask[:, ::-1]
                itw_mask = cv2.warpAffine(itw_mask, itw_img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR).astype(np.float32) > 0
                itw_mask = self.transform(itw_mask.astype(np.float32))
                itw_fg = itw_img * itw_mask; itw_bg = itw_img * (1 - itw_mask);
                itw_mean_fg = torch.sum(itw_fg,(1,2))[:,None,None] / (torch.sum(itw_mask,(1,2))[:,None,None] + 1e-5)
                itw_std_fg = torch.sqrt((torch.sum((itw_fg + (1 - itw_mask) * itw_mean_fg - itw_mean_fg)**2, (1,2)) + 1e-5) / (torch.sum(itw_mask,(1,2)) + 1e-5))[:,None,None]

                #img = itw_bg * (1 - mask) + ((fg - mean_fg) / std_fg * itw_std_fg + itw_mean_fg) * mask
                
                #is_invalid = ((mask == 0) * (itw_mask == 1)).float()
                #img = img * (1 - is_invalid) + itw_fg * is_invalid

                mix_ratio = random.uniform(0.0, 1.0)
                img = itw_bg * (1 - mask) + (itw_fg * mix_ratio + ((fg - mean_fg) / std_fg * itw_std_fg + itw_mean_fg) * (1 - mix_ratio)) * mask

                #cv2.imwrite(str(idx) + '.jpg', img.numpy().transpose(1,2,0)[:,:,::-1]*255)

                """ 
                # mixup
                itw_data = self.itw_ann_list[random.randint(0, len(self.itw_ann_list)-1)]
                itw_img_path, itw_img_shape, itw_bbox = itw_data['img_path'], itw_data['img_shape'], itw_data['bbox']
                itw_img = load_img(itw_img_path)
                #itw_img, _, _, _, _ = augmentation(itw_img, itw_bbox, self.data_split)
                xmin = random.randint(0,itw_img.shape[1] - cfg.input_img_shape[1])
                ymin = random.randint(0,itw_img.shape[0] - cfg.input_img_shape[0])
                itw_img = itw_img[ymin:ymin+cfg.input_img_shape[0], xmin:xmin+cfg.input_img_shape[1], :]
                itw_img = self.transform(itw_img.astype(np.float32))/255.
                #mix_ratio = random.uniform(0.0, 0.5)
                mix_ratio = random.uniform(0.0, 0.25)
                img = img * (1 - mix_ratio) + itw_img * mix_ratio
                #cv2.imwrite(str(random.randint(1,500)) + '.jpg', img.numpy().transpose(1,2,0)[:,:,::-1]*255)
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
            targets = {'joint_img': joint_img, 'smpl_joint_img': smpl_joint_img, 'joint_cam': joint_cam, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(True)}#, 'mask': mask}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bbox': bbox}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # h36m joint from gt mesh
            joint_gt = annot['joint_cam'] 
            joint_gt = joint_gt - joint_gt[self.joint_set['body']['root_joint_idx'],None] # root-relative 
            joint_gt = joint_gt[self.joint_set['body']['eval_joint'],:] 
            
            # h36m joint from param mesh
            mesh_out = out['smpl_mesh_cam'] * 1000 # meter to milimeter
            joint_out = np.dot(self.joint_set['body']['smpl_regressor'], mesh_out) # meter to milimeter
            joint_out = joint_out - joint_out[self.joint_set['body']['root_joint_idx'],None] # root-relative
            joint_out = joint_out[self.joint_set['body']['eval_joint'],:]
            joint_out_aligned = rigid_align(joint_out, joint_gt)
            eval_result['mpjpe'].append(np.sqrt(np.sum((joint_out - joint_gt)**2,1)).mean())
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((joint_out_aligned - joint_gt)**2,1)).mean())

            vis = False
            if vis:
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite(filename + '.jpg', img)
                save_obj(mesh_out, smpl.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
