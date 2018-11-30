import os
from datetime import datetime
import torch


ids_size = 16
d_model = 512
hidden_size = 256
n_classes = 4+1+1
d_ff = 2048
N = 2
n_heads = 16
n_epoch = 2
batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# relative path
place_of_data = '../data'


##########################
# don't touch under script
##########################

class EmitPath:
    def __init__(self, relative_path):
        path_of_here = os.path.dirname(os.path.abspath(__file__))
        self.abs_path = os.path.join(path_of_here, relative_path)
        self.temp_model_path = None
        self.abs_x_path = None
        self.abs_y_path = None
        self.abs_pad_x_path = None
        self.abs_input_model_path = None

    def emit_x_path(self):
        if self.abs_x_path:
            return self.abs_x_path
        return os.path.join(self.abs_path, 'Arithmetic_x.csv')

    def emit_y_path(self):
        if self.abs_y_path:
            return self.abs_y_path
        return os.path.join(self.abs_path, 'Arithmetic_y.csv')

    def emit_pad_x_path(self):
        if self.abs_pad_x_path:
            return self.abs_pad_x_path
        return os.path.join(self.abs_path, 'Arithmetic_pad_x.csv')

    def emit_model_path(self):
        return os.path.join(self.abs_path, 'classify.pt')

    def emit_temp_model_path(self, epochs):
        if self.temp_model_path:
            return os.path.join(self.temp_model_path, 'classify_'+ str(epochs) +'.pt')
        else:
            return os.path.join(self.abs_path, 'classify_'+ str(epochs) +'.pt')

    def emit_losses_path(self):
        return os.path.join(self.abs_path, 'losses.csv')

    def emit_input_model_path(self):
        if self.abs_input_model_path:
            return self.abs_input_model_path
        raise Exception('input_model_path is None')
    
# init EmitPath
emit_path = EmitPath(place_of_data)

x_path = emit_path.emit_x_path
y_path = emit_path.emit_y_path
pad_id_path = emit_path.emit_pad_x_path
model_path = emit_path.emit_model_path
temp_model_path = emit_path.emit_temp_model_path
losses_path = emit_path.emit_losses_path
input_model_path = emit_path.emit_input_model_path
