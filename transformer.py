import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# !모델 정의하기

class TransformerModel(nn.Module):

    def __init__(self, ntoken, emsize, nhead, nhid, nlayers, dropout=0.5):
        """
        ntoken : 
        emsize : embedding size

        """
        # super(!class!, !instance!).__init__()
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, emsize)
        self.emsize = emsize
        self.decoder = nn.Linear(emsize, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
    
        src = self.encoder(src) * math.sqrt(self.emsize)
        # why sqrt?
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # unsqueeze(dim) : returns a new tensor with a dim of size one (at dim position)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# !데이터 로드하고 배치 만들기

import torchtext
from torchtext.data.utils import get_tokenizer
#TEXT = torchtext.data.Field(
TEXT = torchtext.data.ReversibleField(
    tokenize=get_tokenizer("basic_english"),
    init_token='<sos>',
    eos_token='<eos>',
    lower=True
)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    origin_data = data.examples[0].text
    data = TEXT.numericalize([data.examples[0].text])
    #import ipdb; ipdb.set_trace()
    #aa = TEXT.reverse(data)
    # 데이터셋을 bsz 파트들로 나눕니다.
    nbatch = data.size(0) // bsz
    
    data = data.narrow(0, 0, nbatch * bsz)

    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def denumericalize(batch, isTrim=False):
    with torch.cuda.device_of(batch):
        batch = batch.tolist()
    batch = [[TEXT.vocab.itos[ind] for ind in ex] for ex in batch]

    #print([' '.join(ex) for ex in batch])

    if isTrim:
        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, TEXT.eos_token) for ex in batch]

    return [' '.join(ex) for ex in batch]



batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

# backpropagation through time
bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target_batch = source[i+1:i+1+seq_len]
    target = target_batch.view(-1)
    return data, target, target_batch

# !인스턴스 초기화하기

ntokens = len(TEXT.vocab.stoi)  # number of vocab
emsize = 200   # embedding size
nhid = 200   
nlayers = 2    # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
nhead = 2      # 멀티헤드 어텐션(multi-head attention) 모델의 헤드 개수
dropout = 0.2 
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# !모델 실행하기
criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train()  # 학습 모드를 시작합니다.
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets, targets_b = get_batch(train_data, i)  # (35, 20) / (700)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            
            # output tokens
            #amax_i = torch.argmax(output, dim=-1).t().contiguous()  # (20, 35)
            #print(denumericalize(amax_i)[1:3])
            #print(denumericalize(targets_b.t().contiguous()[1:3]))

            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source, verbose=False):
    eval_model.eval() # 평가 모드를 시작합니다.
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
            data, targets, targets_b = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)

            if batch < 3:
                # output tokens
                amax_i = torch.argmax(output, dim=-1).t().contiguous()  # (20, 35)
                print("pred : ", denumericalize(amax_i)[1:2][0])
                print("targ : ", denumericalize(targets_b.t().contiguous())[1:2][0])

            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # 에포크 수
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

### ref 
# scheduler = ...
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
