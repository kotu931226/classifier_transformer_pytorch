import os
import csv
import random
import numpy as np
import torch
from tensorflow.python.lib.io import file_io

class DataOperat:
    @classmethod
    def emit_abspath(cls, path):
        script_dir_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir_path, path)

    @classmethod
    def save_csv(cls, path, items):
        with file_io.FileIO(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(items)

    @classmethod
    def add_csv(cls, path, items):
        if not file_io.file_exists(path):
            with file_io.FileIO(path, 'w') as f:
                f.write("")
        with file_io.FileIO(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(items)

    @classmethod
    def load_csv(cls, path):
        with file_io.FileIO(path, 'r') as f:
            for l in f.read().splitlines():
                yield [w for w in l.split(',')]

    @classmethod
    def save_torch_model(cls, path, model):
        torch.save(model.state_dict(), 'temp_model.pt')
        with file_io.FileIO('temp_model.pt', 'rb') as reader:
            with file_io.FileIO(path, 'wb') as writer:
                writer.write(reader.read())

    @classmethod
    def load_torch_model(cls, path, model):
        with file_io.FileIO(path, 'rb') as reader:
            with file_io.FileIO('temp_model.pt', 'wb') as writer:
                writer.write(reader.read())
        model.load_state_dict(torch.load('temp_model.pt'))
            

    @classmethod
    def create_data_set(cls, pad_id_path, y_path, device='cpu'):
        # create data set from path
        total_size_pad_id = sum(1 for _ in cls.load_csv(pad_id_path))
        total_size_y = sum(1 for _ in cls.load_csv(y_path))
        # if total_size_pad_id != total_size_y:
            # raise Exception(f'there are diffrent size {total_size_pad_id},{total_size_y}')
        gen_pad_id = cls.load_csv(pad_id_path)
        gen_y = cls.load_csv(y_path)
        data_set = []
        for pad_id, y in zip(gen_pad_id, gen_y):
            tmp_pad_id = torch.tensor(list(map(int, pad_id)), dtype=torch.long, device=device)
            tmp_y = torch.tensor(list(map(int, y))[0], dtype=torch.long, device=device)
            data_set.append([tmp_pad_id, tmp_y])
        return data_set

    @classmethod
    def gen_batch_data(cls, data_set, batch_size):
        # genelate data(x, y)
        random.shuffle(data_set)
        total_size = len(data_set)
        # if (total_size // batch_size) < 1:
            # raise Exception(f'batch_size{batch_size} is bigger than total_size{total_size}')
        for i in range(total_size // batch_size):
            yield (torch.stack([data[0] for data in data_set[i*batch_size:(i+1)*batch_size]]),
                   torch.stack([data[1] for data in data_set[i*batch_size:(i+1)*batch_size]]))

class LangOperat:
    @classmethod
    def gen_word2id(cls, word_list, word2id=None):
        # genelate id for word
        if word2id is None:
            word2id = {}
            word2id['PAD'] = 0
            word2id['UNK'] = 1
            word2id['SOS'] = 2
            word2id['EOS'] = 3
            word2id[''] = 4
        
        for w in word_list:
            if w not in word2id.values():
                word2id[w] = len(word2id)
        return word2id

    @classmethod
    def creat_id2word(cls, word2id):
        return {v:k for k, v in word2id.items()}

    @classmethod
    def encode_word2id(cls, word_list, word2id, is_eos=True):
        # atenttion for EOS
        id_list = []
        for word in word_list:
            if word in word2id:
                id_list.append(word2id[word])
            else:
                id_list.append(word2id['UNK'])
        if is_eos:
            id_list.append(word2id['EOS'])
        return id_list

    @classmethod
    def encode_padding_id(cls, id_list, word2id, padding_len=8):
        # zero padding
        if len(id_list) < padding_len:
            while len(id_list) < padding_len:
                id_list.append(word2id['PAD'])
        # elif padding_len < len(id_list):
            # raise Exception(f'id_list is big: {id_list}[{len(id_list)}]')
        return id_list
