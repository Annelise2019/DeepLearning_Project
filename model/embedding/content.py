import torch.nn as nn

class ContentEmbedding(nn.Module):
    
    def __init__(self, feature_size, d_H=512):
        super().__init__()
        
        #N*T,C*V-->nn.linear(feature_size, d_H)
        self.linear = nn.Linear(feature_size, d_H)
        self.embedding_dim = d_H
            
    def forward(self,x):
        N, T, F = x.size()
        x = x.view(N*T,F)
        x = self.linear(x) #N*T,H
        x = x.view(N,T,-1)
                
        return x, self.linear.weight #weight.size()=(C*V,H)
