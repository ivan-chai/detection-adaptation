import torch

import pytorch_lightning as pl

import argparse

from models import LightningDetectorModule, ResnetFPN
from datasets.widerface import LightningDataModule, Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet", type=int, default=18, help="resnet depth (18, 34, 50, 101)")
    parser.add_argument("--epochs", type=int, default=1, help="epochs for training")
    parser.add_argument("--max-lr", type=float, default=3e-2, help="max lr for 1-cycle schedule")
    parser.add_argument("--dir", type=str, default="WIDERFACE", help="path to dataset dir")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--precision", type=int, default=32, help="precision, 32 or 16")
    args = parser.parse_args()
    assert args.resnet in [18, 34, 50, 101]
    assert args.precision in [16, 32]

    return args

if __name__=="__main__":
    args = parse_args()
    pl_model = LightningDetectorModule(ResnetFPN({"depth": args.resnet, "n_points": 5}), {"loss_config": {"landmarks_available": True}})
    pl_data = LightningDataModule({"root": args.dir, "batch_size": args.batch_size, "collate_config": {"grid_h": 2, "grid_w": 2}})

    pl_model.scheduler = lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.max_lr, base_momentum=0, max_momentum=0, steps_per_epoch=12880//args.batch_size, epochs=args.epochs,
            )

    trainer = pl.Trainer(gpus=1, val_check_interval=.1, max_epochs=args.epochs, precision=args.precision)

    trainer.fit(pl_model, datamodule=pl_data)

    ev = Evaluator(args.dir, pl_model)
    metrics = ev.AP_by_difficulty()
    metrics_no_large_faces = ev.AP_by_difficulty(max_face_height=200)

    diffs = ["easy", "medium", "hard"]
    print("AP@50 on WIDERFACE val split:")
    print(" ".join([f" {diff}: {metrics[diff]['AP']:.4f}" for diff in diffs]))
    print("... ignoring large faces:")
    print(" ".join([f" {diff}: {metrics_no_large_faces[diff]['AP']:.4f}" for diff in diffs]))
