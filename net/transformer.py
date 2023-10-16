import torch
import torch.nn as nn
from utils.util import makeSeqMask, makePadMask
from utils.posencode import PositionalEncoding
from net.sublayer import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, device, len_CNvocabulary, len_ENvocabulary, d_model, dff, n_head, n_layer):
        super(Transformer, self).__init__()
        self.len_vocabulary = len_ENvocabulary
        self.device = device
        self.encoder = Encoder(device, len_CNvocabulary, d_model, dff, n_head, n_layer)
        self.decoder = Decoder(device, len_ENvocabulary, d_model, dff, n_head, n_layer)
        self.Projection = nn.Linear(d_model, len_ENvocabulary)      
    
    def forward(self, encoder_input, decoder_input):
        """
        encoder_input.shape -> [batch_size, len_sen1]
        decoder_input.shape -> [batch_size, len_sen2]
        final_output.shape -> [batch_size*len_sen2, len_vocabulary]
        """
        output1 = self.encoder(encoder_input)
        output2 = self.decoder(decoder_input, encoder_input, output1, output1)
        final_output = self.Projection(output2).view(-1, self.len_vocabulary)
        return final_output

class Encoder(nn.Module):
    def __init__(self, device, len_vocabulary, d_model=512, dff=2048, n_head=8, n_layer=6):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention()  # 多头注意力
        self.posembedding = PositionalEncoding()  # 位置编码
        self.input_Embedding = torch.nn.Embedding(len_vocabulary, d_model)  # 词嵌入
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.device = device
        self.FeedForward = torch.nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(True),  
            nn.Linear(dff, d_model)
        )

    def forward(self, encoder_input):
        """
        x.shape -> [batch_size * len_sen * d_model]
        output2.shape -> [batch_size, len_sen, d_model]
        """
        encoder_mask = makePadMask(encoder_input, encoder_input).unsqueeze(-1).repeat(1,1,1,self.n_head)
        x = self.input_Embedding(encoder_input)
        x = self.posembedding(x)
        # 因为是自注意力机制，所以输入MultiHead的QKV都是x,需要循环6次
        for i in range(6):
            output1 = self.MultiHeadAttention(x, x, x, encoder_mask) # shape -> [batch_size, len_sen, n_head, d_v]
            output1 = nn.LayerNorm(self.d_model).to(self.device)(output1+x)
            # FeedForward Layer 后归一化
            output2 = self.FeedForward(output1)
            output2 = nn.LayerNorm(self.d_model).to(self.device)(output1+output2)  # 进行归一化和残差连接
            x = output2
        return x

class Decoder(nn.Module):
    def __init__(self, device, len_vocabulary, d_model=512, dff=2048, n_head=8, n_layer=6):
        super(Decoder, self).__init__()
        self.n_head = n_head
        self.device = device
        self.n_layer = n_layer
        self.MultiHeadAttention = MultiHeadAttention()  # 多头注意力机制
        self.posembedding = PositionalEncoding()  # 位置编码
        self.input_Embedding = torch.nn.Embedding(len_vocabulary, d_model)  # 词嵌入
        self.d_model, self.dff = d_model, dff
        self.FeedForward = torch.nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(True),
            nn.Linear(dff, d_model)
        )
        
    def forward(self, decoder_input, encoder_input, encoder_out1, encoder_out2):
        """
        decoder_input.shape -> [batch_size, len_seq1]
        encoder_input.shape -> [batch_size, len_seq2]
        encoder_out1.shape = encoder_out2.shape -> [batch_size, len_seq2, d_model] 
        """
        # 第一层 (包括Seq mask and PAD mask)
        pad_mask = makePadMask(decoder_input, decoder_input)
        seq_mask = makeSeqMask(decoder_input).to(self.device)
        x = self.input_Embedding(decoder_input)
        x = self.posembedding(x)
        Layer1_mask = (seq_mask+pad_mask).unsqueeze(-1).repeat(1, 1, 1, self.n_head)
        Layer2_padmask = makePadMask(decoder_input, encoder_input).unsqueeze(-1).repeat(1, 1, 1, self.n_head)
        for i in range(6):  # 重复6次
            output1 = self.MultiHeadAttention(x, x, x, Layer1_mask)
            output1 = nn.LayerNorm(self.d_model).to(self.device)(output1+x)
            # 第二层  (没有Seq mask，只有PAD mask)
            output2 = self.MultiHeadAttention(output1, encoder_out1, encoder_out2, Layer2_padmask)
            output2 = nn.LayerNorm(self.d_model).to(self.device)(output2+output1)
            # 第三层
            output3 = self.FeedForward(output2)
            output3 = nn.LayerNorm(self.d_model).to(self.device)(output3+output2)
            x = output3
        return x

