import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

def clones(module, N):
    # moduleList is important
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ClassifyTransformer(nn.Module):
    def __init__(
            self,
            ids_size,
            n_classes,
            d_model=512,
            d_ff=2048,
            N=6,
            n_heads=8,
            max_len=5000,
            pad_id=0,
            dropout=0.1,
            device='cpu',
            is_classify=True
    ):
        super().__init__()
        self.src_embeddings = Embeddings(ids_size, d_model, pad_id=pad_id)
        self.tgt_embeddings = Embeddings(n_classes, d_model, pad_id=pad_id)
        self.position = PositionalEncoding(d_model, dropout=dropout, max_len=max_len, device=device)
        self.encoder = Encoder(d_model, n_heads, d_ff, N, dropout=dropout, device=device)
        self.decoder = Decoder(d_model, n_heads, d_ff, N, dropout=dropout, device=device)
        self.generater = Generater(d_model, n_classes)
        self.is_classify = is_classify
        if is_classify:
            self.classify_tgt = torch.arange(n_classes, dtype=torch.long, device=device)

    def forward(self, scr_x, tgt_x=None, src_mask=None, tgt_mask=None):
        src_embed_x = self.src_embeddings(scr_x)
        # TODO masking
        src_position_x = self.position(src_embed_x)
        # encode_out = self.encoder(position_x, scr_x)
        encoder_out = self.encoder(src_position_x, src_mask)

        if self.is_classify:
            tgt_embed_x = self.tgt_embeddings(self.classify_tgt.repeat(scr_x.size(0), 1))
        else:
            tgt_embed_x = self.tgt_embeddings(tgt_x)
        tgt_position_x = self.position(tgt_embed_x)
        decoder_out = self.decoder(tgt_position_x, encoder_out, src_mask, tgt_mask)
        classes_x = self.generater(decoder_out)
        return classes_x

class Encoder(nn.Module):
    '''Encoder is repeat EncoderLayer'''
    def __init__(self, d_model, n_heads, d_ff, N, dropout=0.1, device='cpu'):
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, d_ff, dropout=dropout, device=device)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model, device=device)

    def forward(self, x, mask):
        # Pass the input (and mask) throuh each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''Base for Encode'''
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, device='cpu'):
        super().__init__()
        self.norm = LayerNorm(d_model, device=device)
        self.multi_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        trans_x = self.norm(x)
        trans_x = self.multi_attention(trans_x, trans_x, trans_x, mask)
        x = self.dropout(trans_x) + x
        trans_x = self.norm(x)
        trans_x = self.feedforward(trans_x)
        x = self.dropout(trans_x) + x
        return x

class Decoder(nn.Module):
    '''Decoder is repeat DecoderLayer'''
    def __init__(self, d_model, n_heads, d_ff, N, dropout=0.1, device='cpu'):
        super().__init__()
        layer = DecoderLayer(d_model, n_heads, d_ff, dropout=dropout, device=device)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(d_model, device=device)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    '''Base for Decoder'''
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, device='cpu'):
        super().__init__()
        self.norm = LayerNorm(d_model, device=device)
        self.self_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.src_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feedforward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        trans_x = self.norm(x)
        trans_x = self.self_attention(trans_x, trans_x, trans_x, tgt_mask)
        x = self.dropout(trans_x) + x
        trans_x = self.norm(x)
        trans_x = self.src_attention(trans_x, encoder_out, encoder_out, src_mask)
        x = self.dropout(trans_x) + x
        trans_x = self.norm(x)
        trans_x = self.feedforward(trans_x)
        x = self.dropout(trans_x) + x
        return x
        
class LayerNorm(nn.Module):
    def __init__(self, features, device='cpu', variance_epsilon=1e-6):
        super().__init__()
        self.gamma = torch.ones(features, device=device, requires_grad=True)
        self.beta = torch.zeros(features, device=device, requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.variance_epsilon)
        return self.gamma * x + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        n_batches = query.size(0)

        # split something for heads
        query, key, value = [l(x).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        
        # into attention layer
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_k)
        return self.linear(x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.transpose(-2, -1) == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, ids_size, d_model, pad_id=None):
        super().__init__()
        self.lut = nn.Embedding(ids_size, d_model, padding_idx=pad_id)
        self.sqrt_d = math.sqrt(d_model)

    def forward(self, x):
        return self.lut(x) * self.sqrt_d

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float)
        position = torch.arange(0, max_len, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *\
        -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Generater(nn.Module):
    '''this is simple log_softmax'''
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_model*n_classes, n_classes)

    def forward(self, x):
        outputs = self.fc1(x.view(x.size(0), -1)) # b, n_classes, d_model -> b, n_model
        outputs = F.log_softmax(outputs.squeeze(), dim=-1)
        return outputs

#############################
# easy test for understanding
#############################
def test_embedding():
    embed = Embeddings(ids_size, d_model)
    # input is LongTensor
    # and input data is less than ids_size
    x = torch.LongTensor([1, 2, 0])
    inputs = x.repeat(n_batches, 1)
    outputs = embed(inputs)
    return outputs

def test_attention():
    # simple test
    x = torch.FloatTensor([[1, 2, 3, 4], [2, 2, 2, 2], [0, 0, 0, 0]])
    x = x.repeat(5, 1, 1)
    mask = None
    dropout = None

    # use embedding test
    # x = test_embedding()
    # mask = torch.tensor([1, 1, 0]).unsqueeze(0)
    # dropout = nn.Dropout(0.1)

    x, attn = attention(x, x, x, mask, dropout)
    return x, attn

def test_multiheadattention():
    multi_attention = MultiHeadAttention(n_heads, d_model)
    x = test_embedding()
    mask = None
    outputs = multi_attention(x, x, x, mask)
    return outputs

def test_classify_transformer():
    x = torch.LongTensor([1, 2, 0]).repeat(n_batches, 1)
    classify_transformer = ClassifyTransformer(
        ids_size,
        n_classes,
        d_model,
        d_ff,
        N,
        n_heads,
        max_len,
        pad_id=0,
        dropout=0.1,
        device='cpu'
    )
    outputs = classify_transformer(x)
    return outputs

if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    n_batches = 5
    ids_size = 16
    n_classes = 16
    d_model = 8
    n_heads = 2
    d_ff = 16
    N = 3
    max_len = 5000
    # print(test_embedding(), '\n', test_embedding().size())
    # ##### attention
    # print(test_embedding())
    # print(test_attention()[0])
    # print(test_attention()[1])
    # print(test_embedding().size())
    # print(test_attention()[0].size())
    # print(test_attention()[1].size())
    # ##### multi_attention
    # print(test_multiheadattention())
    # ##### whole
    # print(test_classify_transformer())

