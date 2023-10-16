import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_modle=512, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        PE = torch.zeros(max_len, d_modle)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_modle, 2).float() / d_modle * (-math.log(10000.0))).unsqueeze(0)
        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)
        PE = PE.unsqueeze(0)  # 对其维度
        self.register_buffer("PE", PE)  # 相当于结构的常量，不参与梯度运算

    def forward(self, x):
        x = x + self.PE[:, :x.size(1), :]
        return self.dropout(x)
    
# test PosEmbedding
# if __name__ == "__main__":
#     peb = PositionalEncoding(512, 5000)
#     input = torch.ones(2,8,512).long()
#     output = peb(input)
