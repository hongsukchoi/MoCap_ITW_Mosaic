import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
import random

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def style_mix(self, img_feat, is_3D, mask):
        """
        # style mix
        mean = torch.mean(img_feat,(2,3))[:,:,None,None].detach()
        std = torch.sqrt(torch.var(img_feat, (2,3)) + 1e-5)[:,:,None,None].detach()
        batch_size = img_feat.shape[0]
        # augment only h36m samples
        itw_idx = torch.nonzero(is_3D == 0).view(-1)
        if batch_size-len(itw_idx) > 0 and len(itw_idx) > 0:
            random_itw_idx = itw_idx[torch.randint(len(itw_idx), (batch_size-len(itw_idx),))].view(-1)
            idx = torch.zeros((batch_size)).long().cuda()
            idx[is_3D == 1] = random_itw_idx
            idx[is_3D == 0] = itw_idx
            mix_ratio = torch.rand(batch_size).float().cuda()[:,None,None,None]
            mixed_mean = mean * mix_ratio + mean[idx,:,:,:] * (1 - mix_ratio)
            mixed_std = std * mix_ratio + std[idx,:,:,:] * (1 - mix_ratio)
            mixed_img_feat = (img_feat - mean) / std * mixed_std + mixed_mean
            img_feat = img_feat * (1 - is_3D)[:,None,None,None].float() + mixed_img_feat * is_3D[:,None,None,None].float()
        return img_feat
        """

        # style mix (fg and bg)
        is_valid = (((torch.sum(mask, (2,3)) > 0).float() + (torch.sum(1 - mask, (2,3)) > 0).float()) > 0).float()[:,:,None,None]
        img_feat_fg = img_feat * mask
        mean_fg = torch.sum(img_feat_fg,(2,3))[:,:,None,None] / (torch.sum(mask,(2,3))[:,:,None,None] + 1e-5)
        std_fg = torch.sqrt((torch.sum((img_feat_fg + (1 - mask) * mean_fg - mean_fg)**2, (2,3)) + 1e-5) / (torch.sum(mask,(2,3)) + 1e-5))[:,:,None,None]
        img_feat_bg = img_feat * (1 - mask)
        mean_bg = torch.sum(img_feat_bg,(2,3))[:,:,None,None] / (torch.sum(1 - mask,(2,3))[:,:,None,None] + 1e-5)
        std_bg = torch.sqrt((torch.sum((img_feat_bg + mask * mean_bg - mean_bg)**2, (2,3)) + 1e-5) / (torch.sum(1 - mask,(2,3)) + 1e-5))[:,:,None,None]
        batch_size = img_feat.shape[0]
        # augment only h36m samples
        itw_idx = torch.nonzero(is_3D == 0).view(-1)
        itw_idx_with_mask = torch.nonzero(is_3D == 0 * is_valid[:,0,0,0]).view(-1)
        if batch_size-len(itw_idx) > 0 and len(itw_idx_with_mask) > 0:
            random_itw_idx = itw_idx_with_mask[torch.randint(len(itw_idx_with_mask), (batch_size-len(itw_idx),))].view(-1)
            idx = torch.zeros(batch_size).long().cuda()
            idx[is_3D == 1] = random_itw_idx
            idx[is_3D == 0] = itw_idx

            mix_ratio_fg = torch.rand(batch_size).float().cuda()[:,None,None,None]
            mixed_mean_fg = mean_fg * mix_ratio_fg + mean_fg[idx,:,:,:] * (1 - mix_ratio_fg)
            mixed_std_fg = std_fg * mix_ratio_fg + std_fg[idx,:,:,:] * (1 - mix_ratio_fg)
            mixed_img_feat_fg = (img_feat_fg - mean_fg) / std_fg * mixed_std_fg + mixed_mean_fg
            mix_ratio_bg = torch.rand(batch_size).float().cuda()[:,None,None,None]
            mixed_mean_bg = mean_bg * mix_ratio_bg + mean_bg[idx,:,:,:] * (1 - mix_ratio_bg)
            mixed_std_bg = std_bg * mix_ratio_bg + std_bg[idx,:,:,:] * (1 - mix_ratio_bg)
            mixed_img_feat_bg = (img_feat_bg - mean_bg) / std_bg * mixed_std_bg + mixed_mean_bg

            img_feat_fg = img_feat_fg * (1 - is_3D)[:,None,None,None].float() + mixed_img_feat_fg * is_3D[:,None,None,None].float()
            img_feat_bg = img_feat_bg * (1 - is_3D)[:,None,None,None].float() + mixed_img_feat_bg * is_3D[:,None,None,None].float()
            img_feat = (img_feat_fg + img_feat_bg) * is_valid + img_feat * (1 - is_valid)
        return img_feat

    def forward(self, x, mode, meta_info):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        #if mode == 'train':
        #    x = self.style_mix(x, meta_info['is_3D'], meta_info['mask'])
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        
        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")


