import torch
import torch.nn as nn
import torch.nn.functional as F
from models.util import sample_and_group 
import argparse
from functools import partial
from typing import Tuple
import torch
from torch import nn, _assert, Tensor

# Local_op的作用：聚合nsamples个点的特征
# input:[batch_size, npoint, nsample, channel]
# output:[batch_siz, channel, npoint]
class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels,affine=True):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.affine = affine
        self.affine_alpha = nn.Parameter(torch.ones([1,1,1,in_channels]))
        self.affine_beta = nn.Parameter(torch.zeros([1,1,1,in_channels]))


    def forward(self, x):
        # new_points-->[32,512,32,128]
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) ,6是SG中前不进行特征变换，直接使用坐标的维度
        # b=32, n=512, s=32, d=128

        # 学习仿射变换
        if self.affine:
            mean = torch.mean(x, dim=2, keepdim=True)
            std = torch.std((x-mean).reshape(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            x = (x-mean)/(std + 1e-5)
            x = self.affine_alpha*x + self.affine_beta
      
        x = x.permute(0, 1, 3, 2) 
        # x-->[32,512,128,32]
        
        x = x.reshape(-1, d, s)
        # [32*512, 128, 32]

        batch_size, _, N = x.size()
        # batch_size=32*512, N=32
        # 这一步已经把batch化为了采样点特征的变化
        
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        # x-->[32*512, 128, 32]

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x-->[32*512, 128]

        x = x.reshape(b, n, -1).permute(0, 2, 1)
        # x-->[32, 128, 512]
        return x


class Pointlstm(nn.Module):
    def __init__(self, output_channels=40):
        super(Pointlstm, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.gather_local_0 = Local_op(in_channels=128, out_channels=128, affine=False)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256, affine=False)
        self.gather_local_2 = Local_op(in_channels=512, out_channels=512, affine=False)
        self.gather_local_3 = Local_op(in_channels=1024, out_channels=1024, affine=False)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True, bias=True, bidirectional=False)

        self.lstm4 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm5 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm6 = nn.LSTM(input_size=64, hidden_size=256, num_layers=1, batch_first=True, bias=True, bidirectional=False)

        self.lstm7 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm8 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm9 = nn.LSTM(input_size=128, hidden_size=512, num_layers=1, batch_first=True, bias=True, bidirectional=False)

        self.lstm10 = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm11 = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bias=True, bidirectional=False)
        self.lstm12 = nn.LSTM(input_size=256, hidden_size=1024, num_layers=1, batch_first=True, bias=True, bidirectional=False)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # x: batch_size, dimensions-->3, point_number-->1024, [32, 3, 1024]

        # xyz: batch_size, point_number-->1024, dimensions-->3, [32, 1024, 3]
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()

        # 位置编码, 在channel上进行变换
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        # batch_size, point_number-->1024, dimensions-->64, [32, 1024, 64]
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)  
        # new_feature-->[B,npoint,nsample,128]
        feature_0 = self.gather_local_0(new_feature)
        feature0 = feature_0.permute(0, 2, 1)
        # feature-->[32, 512, 128]
        # block_1
        x,_ = self.lstm1(feature0)
        x,_ = self.lstm2(x)
        x,_ = self.lstm3(x)
        feature0 = x + feature0

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature0) 
        feature_1 = self.gather_local_1(new_feature)
        feature1 = feature_1.permute(0, 2, 1)
        x,_ = self.lstm4(feature1)
        x,_ = self.lstm5(x)
        x,_ = self.lstm6(x)
        feature1 = x + feature1

        new_xyz, new_feature = sample_and_group(npoint=128, radius=0.2, nsample=32, xyz=new_xyz, points=feature1) 
        feature_2 = self.gather_local_2(new_feature)
        feature2 = feature_2.permute(0, 2, 1)
        x,_ = self.lstm7(feature2)
        x,_ = self.lstm8(x)
        x,_ = self.lstm9(x)
        feature2 = x + feature2

        new_xyz, new_feature = sample_and_group(npoint=64, radius=0.2, nsample=32, xyz=new_xyz, points=feature2) 
        feature_3 = self.gather_local_3(new_feature)
        feature3 = feature_3.permute(0, 2, 1)
        x,_ = self.lstm10(feature3)
        x,_ = self.lstm11(x)
        x,_ = self.lstm12(x)
        feature3 = x + feature3

        feature3 = feature3.permute(0,2,1)

        x = self.conv_fuse(feature3)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
    
if __name__ == '__main__':
    device = torch.device("cuda")
    data = torch.rand(2,3,1024).to(device)

    model = Pointlstm().to(device)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    out = model(data)
    print(out.shape)
