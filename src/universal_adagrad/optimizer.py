import universal_adagrad.optimizers as optimizers


def get_optimizer(opt_dict, params, train_loader, model_base, loss_function):
    opt_name = opt_dict["name"]
    aux = train_loader.dataset.aux
    # get R
    if opt_dict.get("R"):
        R = opt_dict.get("R")
    elif aux.get("R"):
        R = aux.get("R")
    else:
        R = None

    if opt_name == "SVRG":
        opt = optimizers.SVRG(
            params,
            H=opt_dict.get("H", 1000),
            R=R,
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )
    elif opt_name == "AccSVRG":
        opt = optimizers.AccSVRG(
            params,
            H=opt_dict.get("H", 1000),
            R=R,
            A0=opt_dict.get("A0", None),
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif any(["UniSgd" in opt_name, "UniSgd-AdaGrad" in opt_name]):
        opt = optimizers.UniSGD(
            params,
            R=R,
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            stepsize_rule=opt_dict.get("stepsize_rule", "main"),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif any(["UniFastSgd" in opt_name, "UniFastSgd-AdaGrad" in opt_name]):
        opt = optimizers.UniFastSGD(
            params,
            R=R,
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            stepsize_rule=opt_dict.get("stepsize_rule", "main"),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif any(["UniSvrg" in opt_name, "UniSvrg-AdaGrad" in opt_name]):
        opt = optimizers.UniSVRG(
            params,
            R=R,
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            stepsize_rule=opt_dict.get("stepsize_rule", "new"),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif any(["UniFastSvrg" in opt_name, "UniFastSvrg-AdaGrad" in opt_name]):
        opt = optimizers.UniFastSVRG(
            params,
            R=R,
            A0=opt_dict.get("A0", None),
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
            stepsize_rule=opt_dict.get("stepsize_rule", "new"),
            a_rule=opt_dict.get("a_rule", 3),
            initial_proximal_step=opt_dict.get("initial_proximal_step", False),
            count_cheat=opt_dict.get("count_cheat", False),
        )

    elif opt_name == "AdaSVRG":
        opt = optimizers.AdaSVRG(
            params,
            R=R,
            K=opt_dict.get("K", 3),
            eta=opt_dict.get("eta", None),
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif opt_name == "AdaVRAE":
        opt = optimizers.AdaVRAE(
            params,
            R=R,
            eta=opt_dict.get("eta", None),
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    elif opt_name == "AdaVRAG":
        opt = optimizers.AdaVRAG(
            params,
            R=R,
            eta=opt_dict.get("eta", None),
            train_loader=train_loader,
            model_base=model_base,
            loss_function=loss_function,
            replacement=opt_dict.get("replacement", True),
            steps_per_epoch=opt_dict.get("steps_per_epoch", None),
        )

    # others
    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt
