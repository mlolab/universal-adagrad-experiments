import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from universal_adagrad.datasets import get_dataset
from universal_adagrad.models import get_model


def test_polyhedron():
    # Set seed and device
    # ===================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cpu"
    print(
        "\n------Test of UniSgd with AdaGrad stepsize rule "
        + "running on a polyhedron feasibility problem "
        + "with n = 1000, d = 100, q = 2, R = 5 and batch_size = 128.------"
    )

    print("Running on device: %s" % device)

    # test a polyhedron feasibility problem
    # with n = 1000, d = 100, q = 2, R = 5 and batch_size = 128
    # using UniSgd optimizer with AdaGrad stepsize rule

    opt_test = {"name": "UniSgd", "max_epoch": 100, "R": 5, "stepsize_rule": "AdaGrad"}

    exp_dict = {
        "dataset": "polyhedron_dataset",
        "nb_samples": 1000,
        "d": 100,
        "model": "polyhedron_linear_model",
        "loss_func": "polyhedron_loss",
        "validation_metric": "polyhedron_loss",
        "acc_func": "polyhedron_loss",
        "opt": opt_test,
        "batch_size": 128,
        "R": 5,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": False,
        "use_cache_fstar": True,
        "q": 2,
        "runs": [0],
    }

    # Load Datasets
    # ==================
    train_set = get_dataset(
        dataset_name=exp_dict["dataset"], split="train", datadir="", exp_dict=exp_dict
    )

    train_loader = DataLoader(
        train_set,
        drop_last=True,
        shuffle=True,
        sampler=None,
        batch_size=exp_dict["batch_size"],
    )

    val_set = get_dataset(
        dataset_name=exp_dict["dataset"], split="val", datadir="", exp_dict=exp_dict
    )

    # Load Model
    # ==================
    model = get_model(train_loader, exp_dict, device=device)
    score_list = []
    s_epoch = 0

    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["opt"]["max_epoch"]):
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

        # Report
        print(pd.DataFrame(score_list).tail())

    assert train_loss_dict["train_validation_metric"] < 1e-6
