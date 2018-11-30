import enum
import random
import numpy as np
import config
from utils import DataOperat, LangOperat

def gen_arithmetic(num_digis=3):
    for _ in range(NUM_SAMPLE):
        num_1 = random.randint(1, 10**num_digis-1)
        # operators = enum.Enum('operators', ['+', '-', '*', '/'])
        operators = enum.Enum('operators', ['T'])
        operator = random.choice(list(operators))
        num_2 = random.randint(1, 10**num_digis-1)
        formula = str(num_1) + operator.name + str(num_2)
        result = 0
        if operator.name == '+':
            result = num_1 + num_2
        elif operator.name == '-':
            result = num_1 - num_2
        elif operator.name == '*':
            result = num_1 * num_2
        elif operator.name == '/':
            result = num_1 // num_2
        elif operator.name == 'T':
            # 'T' is test operator
            discriminant = num_1 + num_2
            # discriminant = num_1
            if discriminant % 5 == 0:
                result = 4
            elif discriminant % 3 == 0:
                result = 3
            elif discriminant % 2 == 0:
                result = 2
            else:
                result = 1
        yield list(formula), list(str(result))

if __name__ == "__main__":
    NUM_SAMPLE = 10**5
    NUM_DIGIS = 2
    Arithmetic_x = []
    Arithmetic_y = []
    x_path = config.x_path
    y_path = config.y_path
    pad_id_path = config.pad_id_path

    for formula, result in gen_arithmetic(NUM_DIGIS):
        Arithmetic_x.append(formula)
        Arithmetic_y.append(result)
    DataOperat.save_csv(x_path, Arithmetic_x)
    DataOperat.save_csv(y_path, Arithmetic_y)

    sample_x = DataOperat.load_csv(x_path)
    sample_y = DataOperat.load_csv(y_path)
    for _ in range(3):
        # show sample data
        print(sample_x.__next__())
        print(sample_y.__next__())

    word2id = LangOperat.gen_word2id(list('0123456789T'))
    print(word2id)
    data_list = []
    for formula in DataOperat.load_csv(x_path):
        data_list.append(LangOperat.encode_word2id(list(formula), word2id, is_eos=False))
        data_list.append(LangOperat.encode_padding_id(data_list.pop(), word2id, padding_len=5))
    DataOperat.save_csv(pad_id_path, data_list)
    print(data_list[:10])
    print([len(i) for i in data_list[:10]])
    print(np.array(data_list).shape)
