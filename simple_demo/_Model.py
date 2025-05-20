import torch



class Model(torch.nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.lm_head = torch.nn.Parameter(torch.randn(1, vocab_size, dtype=torch.float16))

    def forward(self, input_ids):
        logits = input_ids.unsqueeze(-1).to(self.lm_head.dtype) @ self.lm_head
        probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == "__main__":
    model = Model(20)
    model(1)