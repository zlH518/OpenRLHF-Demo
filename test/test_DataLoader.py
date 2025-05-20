import torch
from simple_demo import DataLoader



def test_DataLoader():
    batch_size = 2
    max_length = 100
    vocab_size = 1024
    dataLoader = DataLoader(batch_size=batch_size, max_length=max_length, vocab_size=vocab_size)
    
    for i, batch in enumerate(dataLoader):
        assert batch.shape[0] == batch_size, 1 
        assert batch.shape[1] <= max_length
        break  


def test_DataLoader_2():
    pass