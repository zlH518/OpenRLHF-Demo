import torch
import torch.distributed as dist


# dp_group = None
# teacher2student_group = None

# def init_group():
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     assert world_size % 2 == 0

#     student_ranks = list(range(0, world_size // 2))
#     teacher_ranks = list(range(world_size // 2, world_size))

#     global dp_group
#     global teacher2student_group

#     for ranks in [student_ranks, teacher_ranks]:
#         _group = dist.new_group(ranks, backend='nccl')
#         if rank in ranks:
#             dp_group = _group
    
#     for ranks in zip(student_ranks, teacher_ranks):
#         _group = dist.new_group(ranks, backend='nccl')
#         if rank in ranks:
#             teacher2student_group = _group
#     return None