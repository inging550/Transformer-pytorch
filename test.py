from net.transformer import Transformer, Encoder, Decoder
from net.sublayer import MultiHeadAttention
from utils.posencode import PositionalEncoding
# from utils.utils import 
import torch


## test
if __name__ == "__main__":
# # test Encoder Layer
    encoder_input = torch.ones(2, 6).long()
    decoder_input = torch.ones(2,7).long()
    mask = torch.zeros(2,6,6,8)
    encoder = Encoder(10, 512, 2048)
    output1 = encoder(encoder_input, mask)

# # test Decoder Layer
    decoder = Decoder(10)
    output = decoder(decoder_input, encoder_input, output1, output1)
    print(1)