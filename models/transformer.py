import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sklearn.metrics as sm
# from torchstat import stat
import torch.nn.functional as F
from torchsummary import summary
import math, copy, time
from torch.nn import TransformerEncoder, TransformerEncoderLayer


'''Multi Self-Attention: Transformer'''
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=4, att_size=64):
        super().__init__()
        '''
            input_dim: 输入维度, 即embedding维度
            output_dim: 输出维度
            head_num: 多头自注意力
            att_size: QKV矩阵维度
        '''
        self.head_num = head_num
        self.att_size = att_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.key = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.value = nn.Linear(input_dim, head_num * att_size, bias=False)
        self.att_mlp = nn.Sequential(
            nn.Linear(head_num*att_size, input_dim),
            nn.LayerNorm(input_dim)
        ) # 恢复输入维度
        self.forward_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        ) # 变为输出维度
        
    def forward(self, x):
        '''
            x.shape: [batch, patch_num, input_dim]
        '''
        batch, patch_num, input_dim = x.shape
        # Q, K, V
        query = self.query(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 1, 3) # [batch, head_num, patch_num, att_size]
        key = self.key(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 3, 1) # [batch, head_num, att_size, patch_num]
        value = self.value(x).reshape(batch, patch_num, self.head_num, self.att_size).permute(0, 2, 1, 3) # [batch, head_num, patch_num, att_size]
        # Multi Self-Attention Score
        z = torch.matmul(nn.Softmax(dim=-1)(torch.matmul(query, key) / (self.att_size ** 0.5)), value) # [batch, head_num, patch_num, att_size]
        z = z.permute(0, 2, 1, 3).reshape(batch, patch_num, -1) # [batch, patch_num, head_num*att_size]
        # Forward
        z = nn.ReLU()(x + self.att_mlp(z)) # [batch, patch_num, input_dim]
        out = nn.ReLU()(self.forward_mlp(z)) # [batch, patch_num, output_dim]
        return out

class Transformer(nn.Module):
    def __init__(self, input_size, patch_size, category, embedding_dim=512):
        super().__init__()
        '''
            input_size: 总体训练样本的shape
            category: 类别数
            embedding_dim: embedding 维度
        '''
        
        self.patch_num = (input_size[0] // patch_size[0] , input_size[1] // patch_size[1])
        self.all_patchs = self.patch_num[0] * self.patch_num[1]
        self.kernel_size = (input_size[-2]//self.patch_num[0], input_size[-1]//self.patch_num[1])
        print('kernel size is ', self.kernel_size)
        self.stride = self.kernel_size
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0
        )
        # 位置信息
        self.position_embedding = nn.Parameter(torch.zeros(1, self.all_patchs, embedding_dim))
        # Multi Self-Attention Layer
        self.msa_layer = nn.Sequential(
            TransformerBlock(embedding_dim, embedding_dim//2), # 4层多头注意力层，每层输出维度下降 1/2
            TransformerBlock(embedding_dim//2, embedding_dim//4),
            TransformerBlock(embedding_dim//4, embedding_dim//8),
            TransformerBlock(embedding_dim//8, embedding_dim//16)
        )
        # classification
        self.dense_tower = nn.Sequential(
            nn.Linear(self.patch_num[0] * self.patch_num[1] * embedding_dim//16, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, category)
        )

    def forward(self, x):
        '''
            x.shape: [b, c, h, w]
        '''
        x = self.patch_conv(x) # [batch, embedding_dim, patch_num[0], patch_num[1]]
        x = self.position_embedding + x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # [batch, all_patchs, embedding_dim]
        x = self.msa_layer(x)
        x = nn.Flatten()(x)
        x = self.dense_tower(x)
        return x


def transformer_choose(dataset = 'uci'):
    if dataset == 'uci':
        model = Transformer(input_size=(128,9), patch_size=(16, 3), category=6, embedding_dim=512).cuda()
        return model
    elif dataset == 'oppo':
        
        model = Transformer(input_size=(30,77), patch_size=(6, 11), category=17, embedding_dim=512).cuda()
        return model
    elif dataset == 'unimib':
        model = Transformer(input_size=(151,3),patch_size=(15, 3), category=17, embedding_dim=512).cuda()
        return model
    #     if res == False:
    #         model = CNN_UNIMIB()
    #     else:
    #         model = ResCNN_UNIMIB()
    #     return model

def main():
    model = transformer_choose(dataset ='oppo')
    input = torch.rand(1, 1, 30, 77).cuda()
    output = model(input)
    print(output.shape)
    # summary(model, (1, 151, 3))
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

if __name__ == '__main__':
    main()