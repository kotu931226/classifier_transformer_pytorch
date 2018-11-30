from statistics import mean
import argparse
import datetime
import sys
import os
path_of_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_of_here, '../'))
import logging
logging.basicConfig(level=logging.INFO)
import torch
from torch import nn
from trainer.utils import DataOperat
from trainer.model import ClassifyTransformer
from trainer import config

# torch.backends.cudnn.benchmark = True

logging.info('start script')

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
model_path = config.model_path
temp_model_path = config.temp_model_path
losses_path = config.losses_path
input_model_path = config.input_model_path

device = config.device
logging.info('use {}'.format(device))

# set argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output-model', help='absolute dir path',
                    default=config.emit_path.abs_path)
parser.add_argument('--output-temp-model', help='absolute dir path',
                    default=None)
parser.add_argument('--train-data', help='absolute file path',
                    default=None)
parser.add_argument('--test-data', help='absolute file path',
                    default=None)
parser.add_argument('--input-model', help='absolute dir path',
                    default=None)
parser.add_argument('--epochs', help='set epochs',
                    type=int, default=config.n_epoch)
parser.add_argument('--job-dir')
args = parser.parse_args()
config.emit_path.abs_path = args.output_model
config.emit_path.temp_model_path = args.output_temp_model
config.emit_path.abs_pad_x_path = args.train_data
config.emit_path.abs_y_path = args.test_data
config.emit_path.abs_input_model_path = args.input_model
n_epoch = args.epochs

logging.info('config loaded')

# create model
model = ClassifyTransformer(ids_size, n_classes, d_model, d_ff, N, n_heads, device=device)
model = model.to(device)
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003)
data_set = DataOperat.create_data_set(pad_id_path(), y_path(), device=device)
dev_idx = len(data_set)*7//8
loss_fn = nn.NLLLoss(ignore_index=0)

if args.input_model:
    DataOperat.load_torch_model(input_model_path(), model)

# start train def
def train(model, data, optimizer, n_epoch, batch_size, dev_data=None):
    for epochs in range(1, n_epoch+1):
        model.train()
        logging.info('-----Epoch: {}'.format(epochs))
        gen_batch_data = DataOperat.gen_batch_data(data, batch_size)
        for i, batch_data in enumerate(gen_batch_data):
            preds = model(batch_data[0])
            loss = loss_fn(preds, batch_data[1])
            if i % 200 == 0:
                logging.info('Epoch {} iteration {} loss {}'.format(epochs, i, loss.item()))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        test(model, dev_data, batch_size)
        if epochs % 20 == 0:
            DataOperat.save_torch_model(temp_model_path(epochs), model)

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
    logging.info('-----Test Result-----')
    logging.info('loss:{}'.format(mean(losses)))
    logging.info('accuracy:{}'.format(correct / count))
    DataOperat.add_csv(losses_path(),
        [[
            mean(losses),
            correct / count,
            datetime.datetime.now()
        ]])

train_data = data_set[:dev_idx]
dev_data = data_set[dev_idx:]

train(model, train_data, optimizer, n_epoch, batch_size, dev_data=dev_data)

DataOperat.save_torch_model(model_path(), model)
logging.info('end train')
