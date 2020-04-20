import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import comm

__all__ = ['launch']

def _find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def launch(main_func, num_gpus_per_machine, num_machines=-1, machine_rank=0, dist_url=None, args=()):
    """
    :param main_func: a funtion that will be called bt 'main_func(*args)'
    :param num_machines:  the total number of machines
    :param machine_rank:  the rank of this machine
    :param dist_url: url to connect to for distributed training, including protocol
                            e.g. "tcp://127.0.0.1:8686"
    :param args:  arguments passed to main_func
    :return:
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
            daemon=False
        )
    else:
        main_func(*args)


def _distributed_worker(local_rank, main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args):
    assert torch.cuda.is_available(), "CUDA is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)
