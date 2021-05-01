import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

class BERT(nn.Module):
    def __init__(self,
                 in_channels=3,
                 keypoints=25, #--->to do: feature_size, d_H, dropout=0.1
                 dropout_in_embedding=0.1,
                 hidden_size=768, #768 to do: hidden, attn_heads, hidden * 4, dropout
                 n_layers_cla=3,
                 num_att_heads=12, #12
                 dropout_in_transformer=0.1,
                 data_bn=True,
                 **kwargs):    
        super().__init__()
    
        self.feature_size = in_channels*keypoints
    
        #data_bn
        if data_bn == True:
            self.bn = nn.BatchNorm1d(self.feature_size)
    
        #data position embedding for BERT
        self.embedding = BERTEmbedding(self.feature_size,hidden_size,dropout_in_embedding) 
    
        #multi-layers transformer blocks, deep network
        self.transformer_blocks_cla = nn.ModuleList([TransformerBlock(hidden_size, num_att_heads, hidden_size * 4, dropout_in_transformer) for _ in range(n_layers_cla)])
    
    
    def forward(self,x):
        N, C, T, V, M = x.size()       
        
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.bn(x) #N*M, T, V*C ---> to research
        x = x.permute(0,2,1).contiguous()#N*M,T,V*C
    
        #To add the first token
        token = torch.ones(x.size(0),1,x.size(-1)).float().to(x.device)
        x = torch.cat((token,x),1)
    
        mask_new = torch.ones(x.size()[:2]).to(x.device)            
        mask_att = (mask_new>0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1) 
    
        #Embedding  x.size()=N*M,T,C*V
        x = self.embedding(x) # embedding the indexed sequence to sequence of vectors  
    
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks_cla:
            x, _ = transformer.forward(x, mask_att) #x.size()=(N*M,T,H)
        
        return x
    
