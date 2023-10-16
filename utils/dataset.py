import torch
from torch.utils.data import Dataset

class Mydata(Dataset):
    def __init__(self, encoder_input, decoder_input, target):
        super(Mydata, self).__init__()
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.target = target
    
    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.target[idx]

    def __len__(self):
        return len(self.encoder_input)