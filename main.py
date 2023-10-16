import argparse
import torch
from torch.utils.data import DataLoader
from utils.dataset import Mydata
from utils.util import make_data, make_Pad
from net.transformer import Transformer

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dff', type=int, default=2048)
    args = parser.parse_args()
    return args

def train(args, dataloader, len_CNvocabulary, len_ENvocabulary):
    # 定义网络，损失函数以及优化器
    net = Transformer(args.device, len_CNvocabulary, len_ENvocabulary, args.d_model, args.dff, args.n_head, args.n_layer)
    net = net.to(args.device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # 去除PAD的影响
    criterion = criterion.to(args.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)

    # 开始训练
    net.train()
    for epoch in range(args.epochs):
        for encoder_input, decoder_input, target in dataloader:
            encoder_input,decoder_input,target = encoder_input.to(args.device),decoder_input.to(args.device),target.to(args.device)
            output = net(encoder_input, decoder_input)
            loss = criterion(output, target.view(-1))
            print("epoch:{}, loss:{}".format(epoch,loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(), 'transformer1.pth')


def pred(args, encoder_input, start_symbol, end_symbol):
    """
    start_symbol 代表 S
    end_symbol 代表 E
    """
    # 定义网络 并导入模型参数
    net = Transformer(args.device, len_CNvocabulary, len_ENvocabulary, args.d_model, args.dff, args.n_head, args.n_layer)
    net = net.to(args.device)
    net.load_state_dict(torch.load("transformer1.pth"))
    # 要从S开始一个字一个字地预测直到E结束
    decoder_input = torch.ones(1, 1).int().to(args.device) * start_symbol # 最开始decoder_input就是 “S”
    while True:  # 无限循环，直到网络输出END字符为止
        encoder_output = net.encoder(encoder_input)
        decoder_output = net.decoder(decoder_input, encoder_input, encoder_output, encoder_output)
        output = net.Projection(decoder_output).view(-1, len_ENvocabulary)
        _, index = output.max(1)  # 概率最大的为下一个字
        next_symbol = torch.ones(1, 1).int().to(args.device) * index[-1]
        if next_symbol == end_symbol:
            break
        else:
            # 要将预测的字放入，再输入到网络
            decoder_input = torch.cat([decoder_input, next_symbol], dim=1)
    return decoder_input


if __name__ == '__main__':
    args = get_args_parser()
    # # 为了方便理解，这里手动定义训练集1
    sentences = [
        ["你 今 晚 回 家 吃 饭 吗 P", "S Are you going home for dinner tonight", "Are you going home for dinner tonight E"],
        ["我 今 晚 回 家 吃 饭 P", "S I'm going home for dinner tonight", "I'm going home for dinner tonight E"],
        ["我 今 晚 不 回 家 吃 饭 P", "S I'm not going home for dinner tonight", "I'm not going home for dinner tonight E"]
    ] 
    test_sentence = [
        ["我 今 晚 回 家 吃 饭 P", "", ""]
    ]

    # 建立中英词库
    CN_glossary = {'P':0,'你':1,'今':2,'饭':3,'家':4,'回':5,'吃':6,'晚':7,'吗':8,'我':9,'不':10}
    EN_glossary = {'P':0,'S':1,'E':2,"I'm":3,'Are':4,'going':5,'you':6,'home':7,
                   'dinner':8,'for':9,'tonight':10, 'not':11}
    CN_idx2word = {i: w for i, w in enumerate(CN_glossary)}
    EN_idx2word = {i:w for i,w in enumerate(EN_glossary)}
    len_CNvocabulary = len(CN_glossary)
    len_ENvocabulary = len(EN_glossary)

    # 数据集2
    sentences1 = [
        ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
        ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
        ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
    ]  
    test_sentence1 = [
        ['我 有 零 个 男 朋 友 P', '', '']
    ]
    CN_glossary1 = {'P': 0, '我': 1, '有': 2, '一': 3, '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
    CN_idx2word1 = {i: w for i, w in enumerate(CN_glossary1)}

    EN_glossary1 = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
                    'friend': 5, 'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11}
    EN_idx2word1 = {i: w for i, w in enumerate(EN_glossary1)}
    len_CNvocabulary1 = len(CN_glossary1)
    len_ENvocabulary1 = len(EN_glossary1)
    
    # 定义trainset以及trainloader
    encoder_input, decoder_input, target = make_data(sentences, CN_glossary, EN_glossary)
    test_encoder_input, _, _ = make_data(test_sentence, CN_glossary, EN_glossary)
    test_encoder_input = torch.LongTensor(test_encoder_input).to(args.device)

    trainset = Mydata(encoder_input, decoder_input, target)
    trainloader = DataLoader(trainset, batch_size=3, shuffle=False, collate_fn=make_Pad)

    # 开始训练
    # train(args, trainloader, len_CNvocabulary, len_ENvocabulary)

    # # 开始预测
    start_symbol, end_symbol = EN_glossary["S"], EN_glossary["E"]
    output = pred(args, test_encoder_input, start_symbol, end_symbol)
    pred_result = [EN_idx2word[i.item()] for i in output[0, :]]
    print("待翻译的句子为： ",test_sentence[0])
    print("网络预测的翻译为： ",pred_result)
