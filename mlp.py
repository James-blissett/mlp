# Code from https://github.com/elcaiseri/Machine-Learning-from-Scratch?source=post_page-----e4cee82ab06d---------------------------------------

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        