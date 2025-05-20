import torch
import torch.distributed as dist


class DistributedAdam(torch.optim.Adam):
    """
    分布式训练优化器，优化的顺序是这样的，梯度清零，然后模型前向，计算loss，再反向传播得到梯度，根据梯度更新参数
    做dp的时候梯度可以分别计算出来，但是更新参数的时候需要把所有的梯度都加起来，所以需要对dp的通信组进行all_reduce
    """
    def __init__(self, *args, group=None, **kwargs):  
        self.group = group
        super().__init__(*args, **kwargs)

    def step(self, closure=None):
        if closure is not None:
            closure().mean().backward()
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, group=self.group)
        super().step()