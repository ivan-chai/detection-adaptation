import torch
from torchvision.ops import nms
import numpy as np

import pytorch_lightning as pl

import argparse

from dalib.config import read_config, prepare_config
from dalib import models
from dalib.models import LightningDetectorModule
from dalib.datasets.widerface import LightningDataModule, Evaluator

import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config-path",
        type=str, required=True,
        help="Path to config file."
    )
    parser.add_argument(
        "-l", "--log-dir",
        type=str, default=os.path.join(os.getcwd(), "lightning_logs"),
        help="Directory where to save logs. If not specified then ./lightning_logs is used."
    )
    parser.add_argument(
        "-n", "--name",
        type=str, default=None,
        help="Experiment name. If not specified then no per-experiment log subdirectory is used."
    )
    parser.add_argument(
        "-g", "--gpus",
        nargs="+", type=int, default=1,
        help="Which GPUs to use."
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str, default="WIDERFACE",
        help="Root directory of the WIDERFACE dataset."
    )
    parser.add_argument(
        "--load",
        type=str, default=None,
        help="If specified, training starts from these model weights."
    )

    args = parser.parse_args()
    return args

def get_default_config():
    return {
        "model": None,
        "data": None,
        "opt": None,
        "seed": 0,
    }

def main(args):
    config = read_config(args.config_path)
    config = prepare_config(get_default_config(), config)
    pl.seed_everything(config["seed"])

    net = getattr(models, config["model"]["net"]["_type"])(config["model"]["net"])
    config["model"].pop("net")
    if args.load:
        pl_model = LightningDetectorModule.load_from_checkpoint(args.load, model=net, config=config["model"])
    else:
        pl_model = LightningDetectorModule(model=net, config=config["model"])

    config["data"]["root"] = args.data_dir
    pl_data = LightningDataModule(config["data"])

    pl_model.scheduler = lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=float(config["opt"]["max_lr"]), div_factor=max(1e-10, float(config["opt"]["max_lr"])/float(config["opt"]["start_lr"])),
                final_div_factor=max(1e-10, float(config["opt"]["max_lr"])/float(config["opt"]["end_lr"])),
                pct_start=float(config["opt"]["pct_start"]),
                base_momentum=float(config["opt"]["base_momentum"]), max_momentum=float(config["opt"]["max_momentum"]), total_steps = int(config["opt"]["steps"]) + 1,
            )

    tb_logger = pl.loggers.TensorBoardLogger(args.log_dir, args.name)
    trainer = pl.Trainer(gpus=args.gpus, val_check_interval=.1, max_steps=int(config["opt"]["steps"]), precision=int(config["opt"]["precision"]), logger=tb_logger)

    trainer.fit(pl_model, datamodule=pl_data)

    ev = Evaluator(args.data_dir, pl_model)
    metrics = ev.AP_by_difficulty()
    metrics_no_large_faces = ev.AP_by_difficulty(max_face_height=200)

    diffs = ["easy", "medium", "hard"]
    print("AP@50 on WIDERFACE val split:")
    print(" ".join([f" {diff}: {metrics[diff]['AP']:.4f}" for diff in diffs]))
    print("... ignoring large faces:")
    print(" ".join([f" {diff}: {metrics_no_large_faces[diff]['AP']:.4f}" for diff in diffs]))
    print()

    for event_name, per_file in ev.pred_and_tar.items():
        for file_name, data in per_file.items():
            scores, bboxes = data["scores"], data["bboxes_pr"]
            scores = torch.tensor(scores) if isinstance(scores, np.ndarray) else scores
            bboxes = torch.tensor(bboxes) if isinstance(bboxes, np.ndarray) else bboxes
            keep = nms(bboxes, scores, 0.5)
            ev.pred_and_tar[event_name][file_name]["scores"] = scores[keep]
            ev.pred_and_tar[event_name][file_name]["bboxes_pr"] = bboxes[keep]

    print("AP@50 on WIDERFACE val split + nms:")
    print(" ".join([f" {diff}: {metrics[diff]['AP']:.4f}" for diff in diffs]))
    print("... ignoring large faces + nms:")
    print(" ".join([f" {diff}: {metrics_no_large_faces[diff]['AP']:.4f}" for diff in diffs]))


if __name__=="__main__":
    main(get_args())
