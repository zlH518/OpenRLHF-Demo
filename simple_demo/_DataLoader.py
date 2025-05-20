import torch



class DataLoader(object):
    def __init__(self, batch_size, max_length, vocab_size):
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def __iter__(self):
        while True:
            length = torch.randint(2, self.max_length,size=(1,))
            # print(length)
            input_ids = torch.randint(0, self.vocab_size, size=(self.batch_size, length), device='cpu')
            yield input_ids