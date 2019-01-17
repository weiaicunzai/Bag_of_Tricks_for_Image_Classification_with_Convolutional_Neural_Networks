
import torch
import torch.nn as nn


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax()
        self.e = e
        self.reduction = reduction
    
    def forward(self, x, target):

        q_hat = (1 - self.e) * target.type(x.type()) + self.e / x.size(1)
        loss = -self.log_softmax(x) * q_hat

        if self.reduction=='mean':
            loss = torch.mean(loss)
        
        elif self.reduction=='sum':
            loss = torch.sum(loss)
        
        elif self.reduction is None:
            pass

        else:
            raise ValueError('wrong reduction value')

        return loss

        
