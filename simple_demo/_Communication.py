import torch
import torch.distributed as dist
from simple_demo._group import teacher2student_group, dp_group

# def send_tensor(tensor, dst):
#     shape = torch.tensor(torch.Size(tensor.size()), dtype=torch.long, device="cuda")
#     ops=[]
#     ops.append(dist.P2POp(op=dist.isend, tensor=shape,peer=dst, group=teacher2student_group))
#     reqs = dist.batch_isend_irecv(ops)
#     for req in reqs:
#         req.wait()
    
#     ops = []
#     ops.append(dist.P2POp(op=dist.isend, tensor=tensor, peer=dst, group=teacher2student_group))
#     reqs = dist.batch_isend_irecv(ops)
#     for req in reqs:
#         req.wait()


# def recv_tensor(src, shape_ndim, dtype):
#     shape = torch.empty(shape_ndim, dtype=torch.long, device="cuda")
#     ops = []
#     ops.append(dist.P2POp(dist.irecv, tensor=shape, peer=src, group=teacher2student_group))
#     reqs = dist.batch_isend_irecv(ops)
#     for req in reqs:
#         req.wait()
    
#     tensor = torch.empty(torch.Size(shape), dtype=dtype, device="cuda")
#     ops = []
#     ops.append(dist.P2POp(dist.irecv, tensor = tensor, peer=src, group=teacher2student_group))
#     reqs = dist.batch_isend_irecv(ops)
#     for req in reqs:
#         req.wait()
#     return tensor
    