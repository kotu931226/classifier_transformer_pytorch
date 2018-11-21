# import random
from statistics import mean
from tqdm import tqdm
import torch
from torch import nn
from utils import DataOperat
from model import ClassifyTransformer
import config

torch.backends.cudnn.benchmark = True

ids_size = config.ids_size
d_model = config.d_model
d_ff = config.d_ff
N = config.N
n_heads = config.n_heads
hidden = config.hidden_size
n_classes = config.n_classes
n_epoch = config.n_epoch
batch_size = config.batch_size
pad_id_path = config.pad_id_path
y_path = config.y_path
save_model_path = config.save_model_path

device = config.device
print('using: ', device)

model = ClassifyTransformer(ids_size, n_classes, d_model, d_ff, N, n_heads, device=device)
model = model.to(device)
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003)
data_set = DataOperat.create_data_set(pad_id_path, y_path, device=device)
dev_idx = len(data_set)*7//8
loss_fn = nn.NLLLoss(ignore_index=0)

def train(model, data, optimizer, n_epoch, batch_size, dev_data=None):
    for i, epoch in enumerate(range(1, n_epoch+1)):
        model.train()
        print('-----Epoch: ', epoch)
        gen_batch_data = DataOperat.gen_batch_data(data, batch_size)
        for j, batch_data in enumerate(tqdm(gen_batch_data)):
            preds = model(batch_data[0])
            loss = loss_fn(preds, batch_data[1])
            if j % 200 == 0 and j != 0:
                print('  Epoch', epoch, 'loss', loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        test(model, dev_data, batch_size)
        if i % 20 == 0 and i != 0:
            torch.save(model.state_dict(), f'./data/temp/classify_{i}.pt')        

def test(model, dev_data, batch_size):
    model.eval()
    correct = 0
    count = 0
    losses = []
    gen_batch_data = DataOperat.gen_batch_data(dev_data, batch_size)
    for batch_data in gen_batch_data:
        preds = model(batch_data[0])
        loss = loss_fn(preds, batch_data[1])
        losses.append(loss.item())
        _, pred_ids = torch.max(preds, 1)
        correct += torch.sum(pred_ids == batch_data[1]).item()
        count += batch_size
    print('-----Test Result-----')
    print('loss:', mean(losses))
    print('accuracy:', correct / count)
    print()
            

train_data = data_set[:dev_idx]
dev_data = data_set[dev_idx:]

train(model, train_data, optimizer, n_epoch, batch_size, dev_data=dev_data)

torch.save(model.state_dict(), save_model_path)
