import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, d_q=64, d_v=64, n_head=8):
        super(MultiHeadAttention, self).__init__()
        self.n_head, self.d_q, self.d_v = n_head, d_q, d_v
        self.W_Q = nn.Linear(d_model, n_head*d_q)
        self.W_k = nn.Linear(d_model, n_head*d_q)
        self.W_V = nn.Linear(d_model, n_head*d_v)
        self.DotProduct = ScaledDotProductAttention()
        self.fc = nn.Linear(n_head * d_v, d_model)
    def forward(self, Q, K, V, mask):
        """
        Q.shape -> [batch_size, len_sen, d_model]
        Q,K 维度要保持一致
        output.shape -> [batch_size, len_sen1, n_head, d_v]
        """
        batch_size, len_sen = Q.shape[0], Q.shape[1]
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_q)  # 这里是先切分了8个子Tensor后输入到线性层
        K = self.W_k(K).view(batch_size, -1, self.n_head, self.d_q)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v) # [batch_size, len_sen, n_head, d_v]
        output = self.DotProduct(Q, K, V, mask) # shape -> [batch_size, len_sen1, n_head, d_v]
        output = self.fc(output.reshape(batch_size, len_sen, -1))
        return output   # reshape将多个子Tensor合并为了最后的输出

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, Q, K, V, mask):
        """
        Q.shape -> [batch_size, len_sen, n_head, d_q]
        mask.shape -> [batch_size, len_sen, len_sen, n_head]
        output.shape -> [batch_size, len_sen1, n_head, d_v]
        """
        scale = torch.einsum('abcd,aecd->abec', Q, K) # shape -> [batch_size, len_sen1, len_sen2, n_head]
        scale.masked_fill_(mask, 1e-9)  # 进行mask
        Q_K = self.softmax(scale)
        output = torch.einsum('abcd,acde->abde', Q_K, V) 
        return output  