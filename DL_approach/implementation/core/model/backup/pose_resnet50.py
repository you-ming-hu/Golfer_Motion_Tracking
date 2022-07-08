import os
import torch
import torch.nn as nn
from collections import OrderedDict

BN_MOMENTUM = 0.1


import core.dataset.common as common
from ..base import DetectionHead

human_keypoints_count = len(common.human_keypoints)
golfclub_keypoints_count = len(common.golfclub_keypoints)

target_count = human_keypoints_count*3 + golfclub_keypoints_count*3 + 5

class Model(torch.nn.Module):
    def __init__(self,pretrained_path):
        super().__init__()
        self.encoder = PoseResNet(pretrained_path)
        self.encoder.inplanes = 2048
        self.heatmap_head = torch.nn.Sequential(
            self.encoder._make_deconv_layer(3,[256, 256, 256],[4, 4, 4]),
            nn.Conv2d(in_channels=256, out_channels=human_keypoints_count + golfclub_keypoints_count, kernel_size=1))
        self.detection_head = DetectionHead(2048,[(3,4),(3,4),(2,4),(2,4),(3,4)])
        
    def intermeidate_forward(self,x):
        feature, multi_people_heatmap = self.encoder(x)
        rest_heatmap = self.heatmap_head(feature)
        detection = self.detection_head(feature)
        return multi_people_heatmap, rest_heatmap, detection
    
    
    def foramt_ouuput(self, multi_people_heatmap, rest_heatmap, detection):
        leading_role_heatmap,golfclub_heatmap = torch.split(rest_heatmap,[human_keypoints_count,golfclub_keypoints_count],dim=1)
    
        leading_role_keypoints,golfclub_keypoints,leading_role_bbox = torch.split(detection,[human_keypoints_count*3,golfclub_keypoints_count*3,5],dim=-1)
        
        leading_role_keypoints_xy, leading_role_keypoints_cf = torch.split(leading_role_keypoints,[human_keypoints_count*2,human_keypoints_count],dim=-1)
        leading_role_keypoints_xy = torch.reshape(leading_role_keypoints_xy,(-1,human_keypoints_count,2))
        
        golfclub_keypoints_xy, golfclub_keypoints_cf = torch.split(golfclub_keypoints,[golfclub_keypoints_count*2,golfclub_keypoints_count],dim=-1)
        golfclub_keypoints_xy = torch.reshape(golfclub_keypoints_xy,(-1,golfclub_keypoints_count,2))
        
        leading_role_bbox_xywh,leading_role_bbox_cf = torch.split(leading_role_bbox,[4,1],dim=-1)
        leading_role_bbox_cf = torch.squeeze(leading_role_bbox_cf,dim=-1)
        
        return {
            'multi_people_heatmap':multi_people_heatmap,
            'leading_role_heatmap':leading_role_heatmap,
            'golfclub_heatmap':golfclub_heatmap,
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf}
            }
    
    def forward(self,x):
        multi_people_heatmap, rest_heatmap, detection = self.intermeidate_forward(x)
        output = self.foramt_ouuput(multi_people_heatmap, rest_heatmap, detection)
        return output
        
    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        multi_people_heatmap, rest_heatmap, detection = self.intermeidate_forward(x)
        detection = torch.sigmoid(detection)
        output = self.foramt_ouuput(multi_people_heatmap, rest_heatmap, detection)
        return output
    
    @torch.no_grad()
    def inference(self,output):
        return {
            'multi_people_heatmap':output['multi_people_heatmap'],
            'leading_role_heatmap':output['leading_role_heatmap'],
            'golfclub_heatmap':output['golfclub_heatmap'],
            'leading_role_keypoints':{'xy':torch.sigmoid(output['leading_role_keypoints']['xy']),'cf':torch.sigmoid(output['leading_role_keypoints']['cf'])},
            'golfclub_keypoints':{'xy':torch.sigmoid(output['golfclub_keypoints']['xy']),'cf':torch.sigmoid(output['golfclub_keypoints']['cf'])},
            'leading_role_bbox':{'xywh':torch.sigmoid(output['leading_role_bbox']['xywh']),'cf':torch.sigmoid(output['leading_role_bbox']['cf'])}
            }
    
    

class PoseResNet(nn.Module):
    def __init__(self,pretrained_path):
        super(PoseResNet, self).__init__()
        
        self.inplanes = 64
        self.deconv_with_bias = False
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(3,[256, 256, 256],[4, 4, 4])
        
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=human_keypoints_count,
            kernel_size=1,
            stride=1,
            padding=0)
        
        self.init_weights(pretrained_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        heatmap = self.deconv_layers(feature)
        heatmap = self.final_layer(heatmap)
        return feature, heatmap 

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError('imagenet pretrained model does not exist')
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
