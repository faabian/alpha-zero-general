import datetime
import json
import getpass
import logging
import os.path
from pathlib import Path
from typing import Optional
import fire
from logging import info

import submitit


from othello.pytorch.NNet import args as model_args
from selfplay import args, selfplay
from utils import dotdict


def get_exp_dir(exp_name: str):
    basepath = Path(f"/checkpoint/{getpass.getuser()}/mcts/")
    if "SLURM_JOB_ID" in os.environ:
        return basepath / exp_name / os.environ["SLURM_JOB_ID"]
    else:
        return (
            basepath
            / exp_name
            / str(datetime.date.today())
            / datetime.datetime.now().strftime("%H-%M-%S")
        )


def launch_generators(args):
    """Adapted from Evariste maxi_gen / adv_gen."""
    exp_dir = get_exp_dir(args.exp_name)
    args = args.to_dict()

    def start_job(job_id: int):
        assert 0 <= job_id < args["n_jobs"]
        dump_dir = exp_dir / "output" / str(job_id)
        try:
            args["seed"] += job_id
            args["checkpoint"] = str(dump_dir)
            selfplay(args=dotdict(args))
        except Exception as e:
            print(e)
            raise e

    # export params

    info(f"Starting launch_generators in {exp_dir}")

    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "params.json", "w") as f:
        json.dump(args, f)

    info(f"Starting {args['n_jobs']} jobs in {exp_dir} ...")

    executor: submitit.Executor
    if args["local"]:
        # assert n_jobs == 1
        # start_job(job_id=0)
        executor = submitit.LocalExecutor(folder=exp_dir)
    else:
        executor = submitit.AutoExecutor(folder=exp_dir, slurm_max_num_timeout=-1)
        executor.update_parameters(
            slurm_job_name=f"parallel_gen__{args['exp_name']}",
            slurm_array_parallelism=2000,
            slurm_partition="learnaccel,devaccel",
            slurm_cpus_per_task=1,
            slurm_gpus_per_task=0,
            slurm_ntasks_per_node=1,
            slurm_mem_gb=60,
            slurm_srun_args=["-vv"],
            slurm_timeout_min=60 * 4,  # in minutes
        )

    jobs = executor.map_array(start_job, range(args["n_jobs"]))

    # results = []
    # for j in jobs:
    #     try:
    #         results.append(j.result())
    #     except Exception as e:
    #         logging.info(f"failed: {str(e)}")
    # return results


if __name__ == "__main__":
    # settings to global variable, no need to pass to launch_generators
    model_args.cuda = False  # no big speedup with GPUs
    launch_generators(args)
