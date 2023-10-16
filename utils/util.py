import torch

# 构造数据
def make_data(sentences, CN_glossary, EN_glossary):
    """
    将单词序列转变为数字序列
    sentenses:待转变的数据集
    """
    encoder_input, decoder_input, target = [], [], []
    max_len = 0
    for i in range(len(sentences)):
        temp = sentences[i][0].split()
        encoder_input.append([CN_glossary[j] for j in temp])
        decoder_input.append([EN_glossary[j] for j in sentences[i][1].split()])
        target.append([EN_glossary[j] for j in sentences[i][2].split()])
    return encoder_input, decoder_input, target

# 对同一batch的数据(中英数据)进行PAD填充
def make_Pad(batch):
    """
    根据dataset取出前batch_size个数据,然后弄成一个列表(Tensor矩阵)
    中英文数据都要进行PAD(填充0)
    """
    batch_size = len(batch)
    maxLen = 3 * [0]
    batch = list(zip(*batch))
    # batch[0]为同一batch下所有的encoder_input
    # batch[1]为-------------的decoder_input
    # batch[2]为-------------的target
    # 现将其PAD，并转成Tensor矩阵 
    
    # 确定同一batch_size下encoder_input,decoder_input,target的最大长度
    for i in range(3):
        for j in range(batch_size):
            maxLen[i] = max(maxLen[i], len(batch[i][j]))
    # 根据最大长度往后补0
    for i in range(3):
        for j in range(batch_size):
            for k in range(len(batch[i][j]), maxLen[i]):
                batch[i][j].append(0)
    return torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2])

# 定义 PAD mask
def makePadMask(input1, input2):
    """
    input1.shape -> [batch_size, len_sen1]
    input2.shape -> [batch_size, len_sen2]
    padmask.shape -> [batch_size, len_sen1, len_sen2]
    默认0代表PAD
    """
    len_sen1 = input1.shape[1]
    padmask = input2.eq(0).unsqueeze(1)  # input2确定mask位置
    padmask = padmask.repeat(1, len_sen1, 1)  # input1确定维度大小，以对其输入
    return padmask

# 定义序列 mask
def makeSeqMask(input):
    """
    input.shape -> [batch_size, len_seq]
    senmask.shape -> [batch_size, len_seq, len_seq]  上三角为1的方阵
    """
    batch_size, len_sen = input.shape[0], input.shape[1]
    senmask = torch.triu(torch.ones(batch_size, len_sen, len_sen), diagonal=1).bool()
    return senmask

