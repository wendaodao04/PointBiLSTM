import numpy as np 
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Pointbilstm_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(Pointbilstm_partseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True, bias=True, bidirectional=True)
        self.conv1_1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1_1 = nn.BatchNorm1d(128)
        
        
        self.conv2_0 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn2_0 = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True, bias=True, bidirectional=True)
        self.conv2_1 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.bn2_1 = nn.BatchNorm1d(256)
        
        self.conv3_0 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.bn3_0 = nn.BatchNorm1d(256)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5, batch_first=True, bias=True, bidirectional=True)
        self.conv3_1 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.bn3_1 = nn.BatchNorm1d(256)
        
        self.conv4_0 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.bn4_0 = nn.BatchNorm1d(256)
        self.lstm4 = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.5, batch_first=True, bias=True, bidirectional=True)
        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        # self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, norm_plt, cls_label):
        # xyz = x.permute(0, 2, 1)
        x = torch.cat([x, norm_plt],dim=1)
        batch_size, _, N = x.size()
        
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x1, _ = self.lstm1(x) # torch.Size([16, 2048, 256])
        # print(x1.shape)
        x1 = x1.permute(0,2,1)
        # print("x1", x1.shape)
        x1 = self.relu(self.bn1_1(self.conv1_1(x1)))  # torch.Size([16, 256, 2048])
        # print("x1*", x1.shape)
        # x1 = x1.permute(0,2,1)

        x2 = self.relu(self.bn2_0(self.conv2_0(x1+x)))
        x2 = x2.permute(0,2,1)
        x2, _ = self.lstm2(x2)
        x2 = x2.permute(0,2,1)
        x2 = self.relu(self.bn2_1(self.conv2_1(x2)))
        # x2 = x2.permute(0,2,1)
        
        x3 = self.relu(self.bn3_0(self.conv3_0(x2)))
        x3 = x3.permute(0,2,1)
        x3, _ = self.lstm3(x3)
        x3 = x3.permute(0,2,1)
        x3 = self.relu(self.bn3_1(self.conv3_1(x3)))
        # x3 = x3.permute(0,2,1)
        
        x4 = self.relu(self.bn4_0(self.conv4_0(x3+x2)))
        x4 = x3.permute(0,2,1)
        x4, _ = self.lstm4(x4)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x = x4.permute(0,2,1)
        
        x = self.conv_fuse(x)
        x_max = torch.amax(x, 2)
        # print(x_max.shape)
        # print(type(x_max))
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
        x = torch.cat((x, x_global_feature), 1) # 1024 * 3 + 64 
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x) # torch.Size([16, 50, 2048])
        # print(x.shape)
        # x = nn.BatchNorm1d(x.size)
        # x = F.log_softmax(x, dim=1)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x
