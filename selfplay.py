import logging

import numpy as np

import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 1,
        "numEps": 10,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 25,  # Number of games moves for MCTS to simulate.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": True,
        "load_folder_file": (
            "./pretrained_models/othello/pytorch/",
            "6x6_153checkpoints_best.pth.tar",
        ),
        # ('/dev/models/8x100x50','best.pth.tar'),
        "seed": 144,  # 44 for train, 144 for eval, 244 for ppl test
        "human": False,  # output MCTS traces for humans or transformer
        "verbose": False,  # print traces
        # for parallel_gen:
        "exp_name": "traces_test",
        "n_jobs": 80,
        "local": False,
    }
)


def selfplay(args):
    log.info(f"Setting seed = {args.seed}")
    np.random.seed(args.seed)
    log.info("Loading %s...", Game.__name__)
    g = Game(6)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    log.info("Starting self-play 🎉")
    c.save_selfplay()


if __name__ == "__main__":
    selfplay(args)
