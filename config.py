import torch

ids_size = 16
d_model = 512
hidden_size = 256
n_classes = 4+1+1
d_ff = 2048
N = 3
n_heads = 16
n_epoch = 20*6
batch_size = 64

x_path = './data/Arithmetic_x.csv'
y_path = './data/Arithmetic_y.csv'
pad_id_path = './data/Arithmetic_pad_x.csv'
save_model_path = './data/classify.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
