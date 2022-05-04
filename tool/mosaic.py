import sys
import os
import os.path as osp
import numpy as np
import cv2
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import Delaunay


sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from utils.vis import vis_keypoints_with_skeleton, vis_keypoints
from utils.preprocessing import load_img
from Human36M.Human36M import Human36M
from MSCOCO.MSCOCO import MSCOCO


MAX_INT = 1000000
SIGMA = 5
JOINT_NUM = 16
SAVE_DIR = '../output/trail2'# '/mnt/prj/sangdoo-yun/Human3.6M/images_greg'
if not osp.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)


def main(subject, part):
    mocap_data = Human36M(subject, part)
    itw_data = MSCOCO()

    save_dir = osp.join(SAVE_DIR, f'Human36M_subject{subject}')
    save_img_dir = osp.join(save_dir, 'images')
    save_meta_dir = osp.join(save_dir, 'meta')
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    if not osp.isdir(save_img_dir):
        os.mkdir(save_img_dir)
    if not osp.isdir(save_meta_dir):
        os.mkdir(save_meta_dir)

    for mocap_idx in tqdm(range(len(mocap_data))):
        # convert mocap joint order to in-the-wild
        mocap_pose = mocap_data.datalist[mocap_idx]['joint_img']
        mocap_pose = transform_joint_to_other_db(mocap_pose, mocap_data.joint_set['body']['joints_name'], itw_data.joint_set['body']['joints_name'])
        mocap_pose_trans_ref = mocap_data.datalist[mocap_idx]['ref_joint_name']
        mocap_pose_trans_ref = transform_joint_to_other_db(mocap_pose_trans_ref, mocap_data.joint_set['body']['joints_name'], itw_data.joint_set['body']['joints_name'])

        # exlcude 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
        mocap_pose = mocap_pose[4:]
        itw_poses = itw_data.pose_list[:, 4:]
        mocap_pose_trans_ref = mocap_pose_trans_ref[4:]
        # change joint name to joint idx
        mocap_pose_trans_ref_idx = []
        for joint_name in mocap_pose_trans_ref:
            idx = itw_data.joint_set['body']['joints_name'].index(joint_name)
            mocap_pose_trans_ref_idx.append(idx)
        mocap_pose_trans_ref_idx = np.array(mocap_pose_trans_ref_idx) - 4

        match_indices, transforms, pose_probs, transformed_poses = get_match_indices(mocap_pose, mocap_pose_trans_ref_idx, itw_poses.copy())

        # x, y
        blend_map = get_mosaic_indices(transformed_poses, pose_probs, itw_data, match_indices, transforms)

        mocap_img2bb_trans = mocap_data.datalist[mocap_idx]['img2bb_trans']
        # mocap_img = load_img(mocap_data.datalist[mocap_idx]['img_path'])
        # mocap_img = cv2.warpAffine(mocap_img, mocap_img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
        # cv2.imshow('mocap', mocap_img / 255.)
        # cv2.waitKey(0)

        # new_img = mocap_img.copy()
        new_img = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1], 3), dtype=np.float32)  # height, width
        # test
        for joint_idx in range(0,JOINT_NUM):
            joint_name = itw_data.joint_set['body']['joints_name'][4+joint_idx]
            # print(joint_name)
            s, R, t = transforms[joint_idx]
            match_idx = match_indices[joint_idx]
            trans = np.concatenate([s*R, t.reshape(-1,1)], axis=1) #np.array([[1, 0, t[0]], [0, 1, t[1]]], dtype=np.float32)
            img_path = itw_data.datalist[match_idx]['img_path']
            img2bb_trans = itw_data.datalist[match_idx]['img2bb_trans']
            img = load_img(img_path)
            img = cv2.warpAffine(img, img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)
            img = cv2.warpAffine(img, trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)

            # new_img[img_idx_map == joint_idx] = img[img_idx_map == joint_idx]
            new_img = new_img + blend_map[:, :, joint_idx:joint_idx+1] * img
            """
            # visualize
            joint_img = itw_data.pose_list[match_idx]
            # affine transformation
            joint_img = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, :1])), 1)
            joint_img[:, :2] = np.dot(trans, joint_img.transpose(1, 0)).transpose(1, 0)
            img = vis_keypoints_with_skeleton(img, joint_img.T, itw_data.joint_set['body']['skeleton'])
            cv2.imshow(f'itw {joint_name}', img / 255.)
            # cv2.waitKey(0)
            tmp = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1], 3), dtype=np.float32)  # height, width
            tmp[img_idx_map == joint_idx] = img[img_idx_map == joint_idx]

            cv2.imshow(f'itw {joint_name} mixed', tmp / 255.)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            """

        # cv2.imshow('mixed', new_img / 255.)
        # cv2.waitKey(0)

        img_name = mocap_data.datalist[mocap_idx]['img_name'].split('/')[-1]
        save_img_path = osp.join(save_img_dir, img_name)
        cv2.imwrite(save_img_path, new_img[:, :, ::-1])
        # save_path = osp.join(SAVE_DIR, img_name + '_mocap.jpg')
        # cv2.imwrite(save_path, mocap_img)
        print(save_img_path)
        save_meta_path = osp.join(save_meta_dir, img_name[:-4] + '.npy')
        np.save(save_meta_path, mocap_img2bb_trans)


def get_mosaic_indices(itw_poses, pose_probs, itw_data, match_indices, transforms):
    xi, yi = np.arange(cfg.input_img_shape[1]), np.arange(cfg.input_img_shape[0])
    xx, yy = np.meshgrid(xi, yi)
    grid_indices = np.stack((xx, yy), axis=-1)

    prob_maps = []
    simplex_found_map = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1]), dtype=np.bool)
    # itw_poses: J x ? x 2
    for joint_idx, pose in enumerate(itw_poses):
        joint_prob_map = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1]), dtype=np.float32)

        valid_indices = np.arange(len(pose))
        valid_indices = np.delete(valid_indices, joint_idx)
        points = np.concatenate([pose[valid_indices][:, :2], pose[joint_idx:joint_idx+1, :2],

                                 ], axis=0)
        pose_prob = np.concatenate([pose_probs[joint_idx], [1.]])

        # add regular grid points for background
        x_space, y_space = cfg.input_img_shape[1] // 4, cfg.input_img_shape[0] // 4
        bg_grid1 = np.stack([grid_indices[0,-1], grid_indices[-1,-1]])
        bg_grid2 = grid_indices[0, ::x_space]
        bg_grid3 = grid_indices[-1, ::x_space]
        bg_grid = np.concatenate([bg_grid1, bg_grid2, bg_grid3])
        bg_point_dist = np.zeros(bg_grid.shape[0])
        for gi in range(bg_point_dist.shape[0]):
            bg_point_dist[gi] = np.min(np.sqrt(np.sum((bg_grid[gi] - points)**2, axis=1)))
        bg_invalid_indices = (bg_point_dist < 30).nonzero()[0]

        bg_grid = np.delete(bg_grid, bg_invalid_indices, axis=0)
        bg_grid_prob = np.zeros_like(bg_grid[:,0])
        points = np.concatenate([points, bg_grid])
        pose_prob = np.concatenate([pose_prob, bg_grid_prob])

        tri = Delaunay(points)

        simplex_map = tri.find_simplex(grid_indices)
        simplex_found_map[simplex_map != -1] = True
        simplices = tri.simplices  # (n simplex, 3)

        for si in range(simplices.shape[0]):
            idx_map = simplex_map == si
            p = grid_indices[idx_map]
            b = tri.transform[si, :2].dot(np.transpose(p - tri.transform[si, 2]))
            b = np.c_[np.transpose(b), 1 - b.sum(axis=0)]

            # joint_bary_map[idx_map] = b
            # v1, v2, v3 = simplices[si]
            # prob_coeff = pose_probs[joint_idx][simplices[si]]  # 3
            prob_coeff = pose_prob[simplices[si]]  # 3

            joint_prob_map[idx_map] = (prob_coeff[None, :] * b).sum(axis=-1)

        prob_maps.append(joint_prob_map)

        """
        # plt.gca().invert_yaxis()
        # plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        # plt.plot(points[:, 0], points[:, 1], 'o')
        # plt.show()
        joint_name = itw_data.joint_set['body']['joints_name'][5+joint_idx]
        cv2.imshow(f'joint prob map {joint_name}', joint_prob_map)
    
        match_idx = match_indices[joint_idx]
        img_path = itw_data.datalist[match_idx]['img_path']
        img2bb_trans = itw_data.datalist[match_idx]['img2bb_trans']
        img = load_img(img_path)
        img = cv2.warpAffine(img, img2bb_trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)

        s, R, t = transforms[joint_idx]
        trans = np.concatenate([s * R, t.reshape(-1, 1)], axis=1)  # np.array([[1, 0, t[0]], [0, 1, t[1]]], dtype=np.float32)
        img = cv2.warpAffine(img, trans, (int(cfg.input_img_shape[1]), int(cfg.input_img_shape[0])), flags=cv2.INTER_LINEAR)

        img = vis_keypoints(img, points, alpha=1)
        # visualize
        cv2.imshow(f'itw image for joint {joint_name}', img / 255.)
        # cv2.waitKey(0)  
        """

    # cv2.waitKey(0)

    prob_maps = np.stack(prob_maps, axis=-1)
    img_idx_map = np.argmax(prob_maps, axis=-1)
    img_idx_map[~simplex_found_map] = -1

    blend_map = np.zeros((cfg.input_img_shape[0], cfg.input_img_shape[1], JOINT_NUM), dtype=np.float32)
    kernel_size = 21

    tmp = np.zeros((cfg.input_img_shape[0]+10*2, cfg.input_img_shape[1]+10*2), dtype=np.float32)
    tmp[10:-10,10:-10] = img_idx_map
    img_idx_map = tmp

    shape = (img_idx_map.shape[0] - kernel_size + 1, img_idx_map.shape[1] - kernel_size + 1, kernel_size, kernel_size)
    strides = 2 * img_idx_map.strides
    patches = np.lib.stride_tricks.as_strided(img_idx_map, shape=shape, strides=strides).reshape(cfg.input_img_shape[0], cfg.input_img_shape[1], -1)

    for ci in range(JOINT_NUM):
        blend_map[:, :, ci] = (patches == ci).sum(axis=2) / (kernel_size ** 2)

    # width = kernel_size // 2
    # for j in range(blend_map.shape[0]):
    #     for i in range(blend_map.shape[1]):
    #         patch = img_idx_map[max(j-width,0):min(j+width+1,cfg.input_img_shape[0]-1), max(i-width,0):min(i+width+1,cfg.input_img_shape[0]-1)]
    #         # for ci in range(JOINT_NUM):
    #         #     blend_map[j, i, ci] = (patch == ci).sum() / (patch.shape[0] * patch.shape[1])
    #
    #         comp, _ = np.histogram(patch, bins=np.arange(JOINT_NUM), density=True)
    #         blend_map[j, i, :len(comp)] = comp

    return blend_map


def get_match_indices(query_pose, query_trans_ref, match_cand_poses):
    # query_pose: J x 2, query_trans_ref: J (indices), match_cand_poses: N x J x 2
    # return J in-the-wild image and pose pairs for one MoCap image

    match_indices = []
    transforms = []
    pose_probs = []
    transformed_poses = []
    for j in range(len(query_pose)):

        B = np.concatenate([query_pose[j,None], query_pose[query_trans_ref[j],None]])
        A = np.concatenate([match_cand_poses[:, j:j+1], match_cand_poses[:, query_trans_ref[j]:query_trans_ref[j]+1]], axis=1)

        c, R, t = batch_rigid_transform(torch.from_numpy(A[:,:,:2]), torch.from_numpy(B[None]))
        cand_poses = torch.from_numpy(match_cand_poses[:, :, :2])
        aligned_cand_poses = torch.bmm(c[:,None,None] * R, cand_poses.permute(0,2,1)).permute(0,2,1) + t

        # compute weights
        valid_indices = np.arange(len(query_pose))
        valid_indices = np.delete(valid_indices, j)

        query_joint_dist = np.sqrt(np.sum((query_pose[j][None] - query_pose)**2,1))  # J
        qjd = query_joint_dist[valid_indices]
        query_joint_weight = 1 / qjd
        query_joint_weight /= np.sum(query_joint_weight)

        cand_joint_dist = torch.sqrt(torch.sum((aligned_cand_poses[:,j:j+1] - aligned_cand_poses) ** 2, 2))  # batch x J
        cjd = cand_joint_dist[:, valid_indices]
        cand_joint_weight = 1 / cjd
        cand_joint_weight /= torch.sum(cand_joint_weight, dim=1, keepdim=True)

        # compute euclidean distance
        euc_dists = torch.sqrt(torch.sum((torch.from_numpy(query_pose[None, valid_indices]) - aligned_cand_poses[:, valid_indices]) ** 2, 2))  # J
        probs = torch.exp(- (euc_dists ** 2) / (SIGMA ** 2))

        weights = torch.from_numpy(query_joint_weight[None]) + cand_joint_weight
        euc_dists = weights * euc_dists
        euc_dists = torch.sum(euc_dists, dim=1)
        euc_dists[torch.isnan(euc_dists)] = MAX_INT

        m_idx = torch.argmin(euc_dists).item()

        match_indices.append(m_idx)
        transforms.append((c[m_idx].numpy(), R[m_idx].numpy(), t[m_idx].numpy()))
        pose_probs.append(probs[m_idx].numpy())
        transformed_poses.append(aligned_cand_poses[m_idx].numpy())

    return match_indices, transforms, pose_probs, transformed_poses


def batch_rigid_transform(A, B):
    centroid_A = torch.mean(A,1,keepdim=True)
    centroid_B = torch.mean(B,1,keepdim=True)
    H = torch.matmul((A - centroid_A).permute(0,2,1), B - centroid_B)

    U, s, V = torch.svd(H)

    R = torch.bmm(V, U.permute(0,2,1))
    # if torch.det(R) < 0:
    #     V = torch.stack((V[:,0],V[:,1],-V[:,2]),1)
    #     R = torch.bmm(V, U.permute(1,0))
    # fix
    invalid_indices = torch.det(R) < 0
    V[invalid_indices] = torch.stack((V[invalid_indices,:,0], -V[invalid_indices,:,1]),2)
    R[invalid_indices] = torch.bmm(V[invalid_indices], U[invalid_indices].permute(0,2,1))

    varP = torch.var(A, dim=1).sum(dim=1)
    c = 1 / varP * torch.sum(s,dim=1)

    t = -torch.matmul(c[:,None,None] * R, centroid_A.permute(0,2,1)).permute(0,2,1) + centroid_B

    return c, R, t


def rigid_transform(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)# A[0] #
    centroid_B = np.mean(B, axis = 0) #B[0] #
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[-1] = -V[-1]  # check if correct
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return (c, R, t)


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=src_joint.dtype)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int)  # 1,5,6,7,8
    parser.add_argument('--part', type=int, default=0)  # 0,1,2,3,4,5,6,7,8,9
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.subject, args.part)