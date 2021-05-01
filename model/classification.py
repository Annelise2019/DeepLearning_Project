import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .two_stream_transformer import BERT


class Classification(nn.Module):

    def __init__(self,
                  n_class,
                  hidden_size=768, 
                  out_feature_size= 12,
                  key_points= 25,
                  dropout_fn = 0.5,
                 **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.pretrainModel = BERT() 
        
        self.fn1 = nn.Linear(self.hidden_size,n_class)
        
        self.dropout = nn.Dropout(dropout_fn)
        
    def forward(self,x):
        output_sequence = self.pretrainModel(x)
        
        N,T,H = output_sequence.size()
        output_sequence = output_sequence.view(-1, 2, T, H) #N,M,T+1,F
        output_sequence = output_sequence.mean(1) #N,T+1,F
        
        video_feature = torch.index_select(output_sequence, index = torch.tensor([0]).to(output_sequence.device),dim=1).squeeze(1) #N,F
        
        video_feature = self.dropout(video_feature)
        output_class = self.fn1(video_feature)  # N x n_class
        
        return output_class
          
    def get_fc_weights(self):
        return self.fn.weight
      
      
