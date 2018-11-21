import torch
from model import ClassifyTransformer
import config
from utils import DataOperat

ids_size = config.ids_size
d_model = config.d_model
d_ff = config.d_ff
N = config.N
n_classes = config.n_classes
n_heads = config.n_heads
device = config.device
y_path = config.y_path
pad_id_path = config.pad_id_path
batch_size = config.batch_size
save_model_path = config.save_model_path

print('test')
model = ClassifyTransformer(ids_size, n_classes, d_model, d_ff, N, n_heads, device=device)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
data_set = DataOperat.create_data_set(pad_id_path, y_path, device)
data_set = data_set[:64*3]

correct = 0
total = 0
for x, y in data_set:
    model.eval()
    output = model(x.unsqueeze(0))
    _, pred_ids = torch.max(output.data, 0)
    total += 1
    correct += 1 if pred_ids == y else 0
    print(x.to('cpu'), int(y), int(pred_ids), output.to('cpu').data)
print(f'Accuracy of all : {correct / total}')

class_correct = list(0. for i in range(n_classes))
class_total = list(0. for i in range(n_classes))
for x, y in data_set:
    output = model(x.unsqueeze(0))
    _, pred_ids = torch.max(output.data, 0)
    correct_ = 1 if int(pred_ids) == int(y) else 0
    for i in range(n_classes):
        y_idx = y
        class_correct[y_idx] += correct_
        class_total[y_idx] += 1

for i in range(n_classes):
    if class_correct[i] < 1 or class_total[i] < 1:
        continue
    print(f'Accuracy of {i} : {class_correct[i]/class_total[i]}')
