import torch
from torch import distributed as dist
import os


def setup_distributed():
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    is_distributed = n_gpu > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    return is_distributed


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def primary():
    if not dist.is_available():
        return True

    if not dist.is_initialized():
        return True

    return get_rank() == 0


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def gather_grad(params):
    world_size = get_world_size()
    
    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def all_gather(input, cat=True):
    if get_world_size() == 1:
        if cat:
            return input
        else:
            return input.unsqueeze(0)
    input_list = [torch.zeros_like(input) for _ in range(get_world_size())]
    synchronize()
    torch.distributed.all_gather(input_list, input, async_op=False)
    if cat:
        inputs = torch.cat(input_list, dim=0)
    else:
        inputs = torch.stack(input_list, dim=0)
    return inputs


def all_gatherv(input, return_boundaries=False):
    """Variable-sized all_gather"""

    # Broadcast the number of elements in every process:
    num_elements = torch.tensor(input.size(0), device=input.device)
    num_elements_per_process = all_gather(num_elements, cat=False)
    max_elements = num_elements_per_process.max()
    # Add padding so every input is the same size:
    difference = max_elements - input.size(0)
    if difference > 0:
        input = torch.cat([input, torch.zeros(difference, *input.size()[1:], device=input.device, dtype=input.dtype)], 0)
    inputs = all_gather(input, cat=False)
    # Remove padding:
    inputs = torch.cat([row[:num_ele] for row, num_ele in zip(inputs, num_elements_per_process)], 0)
    if return_boundaries:
        boundaries = torch.cumsum(num_elements_per_process, dim=0)
        boundaries = torch.cat([torch.zeros(1, device=input.device, dtype=torch.int), boundaries], 0)
        return inputs, boundaries.long()
    else:
        return inputs


def all_reduce(input, device):
    num_local = torch.tensor([input.size(0)], dtype=torch.float, device=device)
    input = input.sum(dim=0, keepdim=True).to(device)
    num_global = all_gather(num_local).sum()
    input = all_gather(input)
    input = input.sum(dim=0).div(num_global)
    return input


def rank0_to_all(input):
    input = all_gather(input)
    rank0_input = input[0]
    return rank0_input


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses
