import os
import time
import torch
import random



import numpy as np
import torch.distributed as dist



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl")
    print(f'Hello World from rank:{torch.distributed.get_rank()} out of {torch.distributed.get_world_size()} worlds.')

    devices = torch.cuda.device_count()
    torch.cuda.set_device(rank % devices)
    
    # all reduce 和all gather测试
    # tensor = torch.tensor([rank+2],dtype=torch.float32, device="cuda")
    # tensor2 = torch.tensor([1.0], dtype=torch.float32, device = "cuda")
    # tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    # torch.distributed.all_gather(tensors, tensor2)
    # torch.distributed.all_reduce(tensor)
    # print(sum(tensors))
    # print(tensor)
    
    # boradcast测试
    # if rank == 0:
    #     tensor = torch.tensor([5], dtype = torch.float32, device="cuda")
    # else:
    #     tensor = torch.tensor([1], dtype = torch.float32, device="cuda")
    #     print(f"rank:{rank}, tensor:{tensor}")
    # torch.distributed.broadcast(tensor, 0)
    # print(f"rank:{rank}, tensor:{tensor}")

    # 测试p2p通信
    # if rank % 2 == 0:
    #     tensor = torch.randn(1,4,dtype=torch.float32, device="cuda")
    #     print(f"tensor:{tensor}")
    #     shape_tensor = torch.tensor(tensor.size(), dtype = torch.long, device="cuda")
    #     dist.send(shape_tensor, rank+1)
    #     dist.send(tensor, rank+1)
    # else:
    #     shape = torch.empty(2, dtype=torch.long, device="cuda")
    #     dist.recv(shape, rank-1)
    #     tensor = torch.empty(torch.Size(shape), dtype=torch.float16, device="cuda")
    #     dist.recv(tensor, rank-1)
    #     print(f"rank:{rank}, shape:{shape}, tensor:{tensor}")

    # 测试barrier
    # if rank == 0:
    #     time.sleep(20)
    # else:
    #     a = 1 + 2
    # dist.barrier()

    # 测试通信组
    ranks = [0, 1]
    group = dist.new_group(ranks)

    if rank in ranks:
        dist.barrier(group=group)