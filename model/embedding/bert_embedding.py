import torch.nn as nn
from .content import ContentEmbedding
from .position import PositionalEmbedding

class BERTEmbedding(nn.Module):
    """
        BERT Embedding which is consisted with under features
        1. ContentEmbedding: normal embedding matrix
        2. PositionEmbedding: adding positional information using sin, cos
        
        sum of all these features are output of BertEmbedding
    """

    def __init__(self, feature_size, d_H, dropout=0.1):
        """
        :param feature_size: feature size of the output of gcn
        :param d_H: embedding size of frame's spatial vector
        :param dropout: dropout rate
        """
        super().__init__()
        self.content = ContentEmbedding(feature_size=feature_size,d_H=d_H)
        self.position = PositionalEmbedding(d_H=self.content.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = d_H
        
        
        
    def forward(self, x, position = True, content = True):
    
    
        if content:
            x, self.content_weight = self.content(x)

        
        if position:
            x = x+self.position(x)
            
        return self.dropout(x) #add embedding weight to return 

    def get_weight(self):
        return self.content_weight
