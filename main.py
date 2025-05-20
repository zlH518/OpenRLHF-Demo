import os
import torch
import torch.distributed as dist
import random
import numpy as np
import logging

from simple_demo import * 


vocab_size=20
batch_size=1
max_length=200
dp_group = None
teacher2student_group = None
log_path='logs'

def init_group():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % 2 == 0

    student_ranks = list(range(0, world_size // 2))
    teacher_ranks = list(range(world_size // 2, world_size))

    global dp_group
    global teacher2student_group

    for ranks in [student_ranks, teacher_ranks]:
        _group = dist.new_group(ranks, backend='nccl')
        if rank in ranks:
            dp_group = _group
    
    for ranks in zip(student_ranks, teacher_ranks):
        _group = dist.new_group(ranks, backend='nccl')
        if rank in ranks:
            teacher2student_group = _group
    return None

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def send_tensor(tensor, dst):
    shape = torch.tensor(torch.Size(tensor.size()), dtype=torch.long, device="cuda")
    ops=[]
    ops.append(dist.P2POp(op=dist.isend, tensor=shape,peer=dst, group=teacher2student_group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    
    ops = []
    ops.append(dist.P2POp(op=dist.isend, tensor=tensor, peer=dst, group=teacher2student_group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


def recv_tensor(src, shape_ndim, dtype):
    shape = torch.empty(shape_ndim, dtype=torch.long, device="cuda")
    ops = []
    ops.append(dist.P2POp(dist.irecv, tensor=shape, peer=src, group=teacher2student_group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    
    tensor = torch.empty(torch.Size(shape), dtype=dtype, device="cuda")
    ops = []
    ops.append(dist.P2POp(dist.irecv, tensor = tensor, peer=src, group=teacher2student_group))
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return tensor
    

def setup_distributed_logger(log_dir=None, name="train", rank=None):
    """初始化分布式日志器"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0

    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    
    # 格式添加 rank 信息
    formatter = logging.Formatter(
        f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s'
    )

    # 控制台输出（所有进程都打印）
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出（每个进程独立文件）
    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



if __name__ == '__main__':
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend='nccl'
    )
    devices = torch.cuda.device_count()
    torch.cuda.set_device(rank % devices)
    init_group()

    assert world_size % 2 ==0
    teacher_offset=world_size//2

    logger = setup_distributed_logger(log_path)
    if rank < teacher_offset:   #student
        """
        student rank: 1.设置随机种子，初始化数据集、模型、优化器
        """
        logger.info("init student rank")
        set_random_seed(1407)
        logger.info("set random seed")
        dataLoader = DataLoader(batch_size=batch_size, max_length=max_length, vocab_size=vocab_size)
        logger.info("init dataLoader")
        model = Model(vocab_size=vocab_size).half().cuda()
        logger.info("init Model")
        optimizer = DistributedAdam(model.parameters(), lr=1e-2,group=dp_group,eps=1e-4)
        logger.info("init optimizer")

        for i, input_ids in enumerate(dataLoader):
            logger.info(f"input_ids shape:{input_ids.shape}")
            if i % teacher_offset != rank:
                continue
            optimizer.zero_grad()
            logger.info("zero grad")
            input_ids = input_ids.cuda()
            send_tensor(input_ids, rank+teacher_offset)
            logger.info("send input")
            student_probs = model(input_ids)
            logger.info("compute probs")

            teacher_probs = recv_tensor(rank+teacher_offset, 3, torch.float16)
            logger.info("get probs from teacher")
            kl_loss = teacher_probs * ((teacher_probs+1e-5).log() - (student_probs+1e-5).log())
            kl_loss = kl_loss.sum(-1).mean() / dist.get_world_size(group=dp_group)
            logger.info("compute loss")
            kl_loss.backward()
            logger.info("loss back")
            optimizer.step()
            logger.info("optimizer weight")

            reporting_loss = kl_loss.clone()
            dist.all_reduce(reporting_loss, group=dp_group)
            logger.info("get all loss")
            print(f"rank: {rank}; loss:{reporting_loss}; kl loss:{kl_loss} weight:{model.lm_head.data[0,:2]}",flush=True)
            dist.barrier(group=dp_group)

            # if i >=30:
            #     break
    else:
        """
        teacher:1.设置随机数，初始化模型；2.接受student的数据，计算probs，传回去
        """
        set_random_seed(1408)
        logger.info("init student rank")
        model = Model(vocab_size=vocab_size).half().cuda()
        model.eval()
        logger.info("init Model")
        while True:
            input_ids = recv_tensor(rank-teacher_offset, 2, torch.int64)
            logger.info("get input")
            teacher_probs = model(input_ids)
            logger.info("compute probs")
            send_tensor(teacher_probs, rank-teacher_offset)
            logger.info("send probs")
