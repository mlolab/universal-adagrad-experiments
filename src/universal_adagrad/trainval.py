import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import universal_adagrad.utils as ut
from universal_adagrad import datasets, models

# cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    # Set seed and device
    # ===================
    seed = 42 + exp_dict["runs"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = "cuda"
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not available please run with "-c 0"'
    else:
        device = "cpu"

    print("Running on device: %s" % device)

    savedir_base = args.savedir_base
    os.makedirs(savedir_base, exist_ok=True)

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(
        dataset_name=exp_dict["dataset"],
        split="train",
        datadir=args.datadir,
        exp_dict=exp_dict,
    )

    train_loader = DataLoader(
        train_set,
        drop_last=True,
        shuffle=True,
        sampler=None,
        batch_size=exp_dict["batch_size"],
    )

    val_set = datasets.get_dataset(
        dataset_name=exp_dict["dataset"],
        split="val",
        datadir=args.datadir,
        exp_dict=exp_dict,
    )

    # Load Model
    # ==================
    model = models.get_model(train_loader, exp_dict, device=device)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = ut.load_pkl(score_list_path)
        model.set_state_dict(torch.load(model_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["opt"]["max_epoch"]):
        # Set seed
        seed = epoch + exp_dict.get("runs", 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Validate one epoch
        train_loss_dict, grad_norm_dict = model.val_on_trainset(
            train_set,
            metric=exp_dict["validation_metric"],
            name="validation_metric",
            record_grad_norm=exp_dict["record_grad_norm"],
        )
        val_acc_dict = model.val_on_testset(
            val_set, metric=exp_dict["acc_func"], name="acc"
        )

        # Record metrics
        score_dict = {"epoch": epoch}
        score_dict.update(train_loss_dict)
        score_dict.update(val_acc_dict)
        if exp_dict["record_grad_norm"]:
            score_dict.update(grad_norm_dict)
        score_dict["nb_forwards"] = model.opt.state.get("nb_forwards", {})
        score_dict["nb_backwards"] = model.opt.state.get("nb_backwards", {})
        score_dict["nb_update_full_grad"] = model.opt.state.get(
            "nb_update_full_grad", {}
        )
        if model.opt.state.get("A") is not None:
            score_dict["A"] = model.opt.state.get("A")
        if epoch > 0:
            if model.opt.state.get("step_size") is not None:
                score_dict["step_size"] = model.opt.state.get("step_size")
            else:
                score_dict["step_size"] = exp_dict["opt"]["lr"]

        # Train one epoch
        s_time = time.time()
        model.train_one_epoch()
        e_time = time.time()

        if epoch == 0:
            score_dict["step_size"] = model.opt.state.get("initial_step_size")

        score_dict["train_epoch_time"] = e_time - s_time

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        ut.save_pkl(score_list_path, score_list)
        if exp_dict["save_model"] is True:
            ut.torch_save(model_path, model.get_state_dict())
        print("Saved: %s" % savedir)
