from haven import haven_utils as hu

"""
model: options include ['svm', 'logistic_regression', 'huber_regression',
                        'L2_regression', 'polyhedron_linear_model', 'resnet...']
loss_func: the loss function used to train the model, 
    options include ['logistic_loss', 'hinge_loss', 'huber_loss', 'L2_loss', 
                                                'softmax_loss', 'polyhedron_loss']
validation_metric: the metric used for validation 
        of the training set after each epoch which is usually the function value gap
    options include ['logistic_func_gap', 'hinge_func_gap', 'huber_func_gap', 
                    'L2_func_gap', 'softmax_accuracy', 'class2_acc', also various loss_func]
acc_func: the accuracy metric used for the validation of the test set which is usually the accuracy.
        (options are the same as validation_metric)
sigma2: L2 regularization coefficient, by default it is None (0)
replacement: sampling with/without replacement
R: radius of the ball centered at origin. If None, then the constraint for the problem is not 
    active except the case of polyhedron feasibility problem where a default R is always used.
record_grad_norm: whether or not record gradient norm for the training dataset 
runs: a list of index of the runs. If the list is [0], then the experiment is run once
batch_size: the batch size used for training 
save_model: whether or not save the model
use_cache_fstar: whether or not use the cached fstar, if False, then fstar is computed from scratch 
        Note for the polyhedron feasibility problem, fstar is always zero

special parameters
mu_hyber: the parameter in the huber loss
nb_samples: the number of samples in the synthetic dataset for L2_regression and polyhedron feasibility problem
d: the dimension of the synthetic dataset for L2_regression
"""
# --------------------------------------------------------------------------------------------
EXP_GROUPS = {}

# polyhedron feasibility problem
# n = 10000, d = 1000, q = 1
opt_polyhedron_nd1 = [
    {"name": "UniFastSgd", "max_epoch": 3300, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 3300,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSgd", "max_epoch": 6600, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniSgd-AdaGrad",
        "max_epoch": 6600,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AccSVRG", "max_epoch": 3400, "R": 1000000, "H": 1 / 100},
    {"name": "UniFastSvrg", "max_epoch": 2500, "R": 1000000},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 2500,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaSVRG", "max_epoch": 17, "R": 1000000},
    {"name": "AdaVRAE", "max_epoch": 2500, "R": 1000000},
    {"name": "AdaVRAG", "max_epoch": 2500, "R": 1000000},
    {"name": "UniSvrg", "max_epoch": 17, "R": 1000000},
    {
        "name": "UniSvrg-AdaGrad",
        "max_epoch": 17,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
]

EXP_GROUPS["polyhedron_nd1"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_polyhedron_nd1,
        "batch_size": 256,
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1,
        "runs": [0],
    }
)


# n = 10000, d = 1000, q = 1.3
opt_polyhedron_nd2 = [
    {"name": "UniSgd", "max_epoch": 4000, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniSgd-AdaGrad",
        "max_epoch": 4000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniFastSgd", "max_epoch": 2000, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 2000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AccSVRG", "max_epoch": 2000, "R": 1000000, "H": 1 / 100},
    {"name": "UniFastSvrg", "max_epoch": 1400, "R": 1000000},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 1400,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSvrg", "max_epoch": 17, "R": 1000000},
    {
        "name": "UniSvrg-AdaGrad",
        "max_epoch": 17,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaVRAE", "max_epoch": 1400, "R": 1000000},
    {"name": "AdaVRAG", "max_epoch": 1400, "R": 1000000},
    {"name": "AdaSVRG", "max_epoch": 14, "R": 1000000},
]

EXP_GROUPS["polyhedron_nd2"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_polyhedron_nd2,
        "batch_size": 256,
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.3,
        "runs": [0],
    }
)

# n = 10000, d = 1000, q = 1.6
opt_polyhedron_nd3 = [
    {"name": "UniSgd", "max_epoch": 2000, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniSgd-AdaGrad",
        "max_epoch": 2000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniFastSgd", "max_epoch": 1000, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 1000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AccSVRG", "max_epoch": 1000, "R": 1000000, "H": 1 / 10},
    {"name": "UniFastSvrg", "max_epoch": 580, "R": 1000000},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 580,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSvrg", "max_epoch": 15, "R": 1000000},
    {
        "name": "UniSvrg-AdaGrad",
        "max_epoch": 15,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaVRAE", "max_epoch": 580, "R": 1000000},
    {"name": "AdaVRAG", "max_epoch": 580, "R": 1000000},
    {"name": "AdaSVRG", "max_epoch": 13, "R": 1000000},
]

EXP_GROUPS["polyhedron_nd3"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_polyhedron_nd3,
        "batch_size": 256,
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.6,
        "runs": [0],
    }
)

# n = 10000, d = 1000, q = 2
opt_polyhedron_nd4 = [
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 220,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniFastSgd", "max_epoch": 220, "R": 1000000, "stepsize_rule": "main"},
    {"name": "UniSgd", "max_epoch": 440, "R": 1000000, "stepsize_rule": "main"},
    {
        "name": "UniSgd-AdaGrad",
        "max_epoch": 440,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AccSVRG", "max_epoch": 220, "R": 1000000, "H": 1},
    {"name": "UniFastSvrg", "max_epoch": 150, "R": 1000000},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 150,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSvrg", "max_epoch": 13, "R": 1000000},
    {
        "name": "UniSvrg-AdaGrad",
        "max_epoch": 13,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaVRAE", "max_epoch": 150, "R": 1000000},
    {"name": "AdaVRAG", "max_epoch": 150, "R": 1000000},
    {"name": "AdaSVRG", "max_epoch": 12, "R": 1000000},
]

EXP_GROUPS["polyhedron_nd4"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_polyhedron_nd4,
        "batch_size": 256,
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 2,
        "runs": [0],
    }
)


# minibatch effect test
opt_minibatch_polyhedron_1 = [
    {
        "name": "UniFastSvrg-AdaGrad-B-64",
        "max_epoch": 1000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_1"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_1,
        "batch_size": [64],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_2 = [
    {
        "name": "UniFastSvrg-AdaGrad-B-512",
        "max_epoch": 700,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_2"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_2,
        "batch_size": [512],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_3 = [
    {
        "name": "UniFastSvrg-AdaGrad-B-4096",
        "max_epoch": 1000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_3"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_3,
        "batch_size": [4096],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)


opt_minibatch_polyhedron_4 = [
    {
        "name": "UniSvrg-AdaGrad-B-64",
        "max_epoch": 14,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_4"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_4,
        "batch_size": [64],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_5 = [
    {
        "name": "UniSvrg-AdaGrad-B-512",
        "max_epoch": 14,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_5"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_5,
        "batch_size": [512],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_6 = [
    {
        "name": "UniSvrg-AdaGrad-B-4096",
        "max_epoch": 15,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_6"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_6,
        "batch_size": [4096],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)


opt_minibatch_polyhedron_7 = [
    {
        "name": "UniFastSgd-AdaGrad-B-64",
        "max_epoch": 1000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_7"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_7,
        "batch_size": [64],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_8 = [
    {
        "name": "UniFastSgd-AdaGrad-B-512",
        "max_epoch": 800,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_8"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_8,
        "batch_size": [512],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_9 = [
    {
        "name": "UniFastSgd-AdaGrad-B-4096",
        "max_epoch": 3000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_9"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_9,
        "batch_size": [4096],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)


opt_minibatch_polyhedron_10 = [
    {
        "name": "UniSgd-AdaGrad-B-64",
        "max_epoch": 150,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_10"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_10,
        "batch_size": [64],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_11 = [
    {
        "name": "UniSgd-AdaGrad-B-512",
        "max_epoch": 1000,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_11"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_11,
        "batch_size": [512],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)

opt_minibatch_polyhedron_12 = [
    {
        "name": "UniSgd-AdaGrad-B-4096",
        "max_epoch": 1500,
        "R": 1000000,
        "stepsize_rule": "AdaGrad",
    }
]
EXP_GROUPS["minibatch_polyhedron_12"] = hu.cartesian_exp_group(
    {
        "dataset": "polyhedron_dataset",
        "nb_samples": 10000,
        "d": 1000,
        "model": ["polyhedron_linear_model"],
        "loss_func": ["polyhedron_loss"],
        "validation_metric": ["polyhedron_loss"],
        "acc_func": ["polyhedron_loss"],
        "opt": opt_minibatch_polyhedron_12,
        "batch_size": [4096],
        "R": 1000000,
        "sigma2": None,
        "record_grad_norm": False,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "q": 1.5,
        "runs": [0],
    }
)


# logistic regression
# mushrooms (L_max = 5.25, solution on the boundary, d << n)
opt_logis_mushrooms = [
    {"name": "UniFastSgd", "max_epoch": 100, "R": 1, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 100,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSgd", "max_epoch": 200, "R": 1, "stepsize_rule": "main"},
    {"name": "UniSgd-AdaGrad", "max_epoch": 200, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniSvrg", "max_epoch": 9, "R": 1},
    {"name": "UniSvrg-AdaGrad", "max_epoch": 9, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniFastSvrg", "max_epoch": 70, "R": 1},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 70,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaSVRG", "max_epoch": 12, "R": 1},
    {"name": "AdaVRAE", "max_epoch": 70, "R": 1},
    {"name": "AdaVRAG", "max_epoch": 70, "R": 1},
    {"name": "SVRG", "max_epoch": 9, "R": 1, "H": 5.25},
    {"name": "AccSVRG", "max_epoch": 100, "R": 1, "H": 5.25},
]

EXP_GROUPS["logis_mushrooms"] = hu.cartesian_exp_group(
    {
        "dataset": "mushrooms",
        "model": ["logistic_regression"],
        "loss_func": ["logistic_loss"],
        "validation_metric": ["logistic_func_gap"],
        "acc_func": ["class2_acc"],
        "opt": opt_logis_mushrooms,
        "R": 1,
        "batch_size": 32,
        "sigma2": None,
        "record_grad_norm": True,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "runs": [0, 1, 2],
    }
)


# w8a (L_max = 28.5, solution on the boundary, d << n)
opt_logis_w8a = [
    {"name": "UniFastSgd", "max_epoch": 80, "R": 1, "stepsize_rule": "main"},
    {"name": "UniFastSgd-AdaGrad", "max_epoch": 80, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniSgd", "max_epoch": 160, "R": 1, "stepsize_rule": "main"},
    {"name": "UniSgd-AdaGrad", "max_epoch": 160, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniSvrg", "max_epoch": 8, "R": 1, "stepsize_rule": "new"},
    {"name": "UniSvrg-AdaGrad", "max_epoch": 8, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniFastSvrg", "max_epoch": 50, "R": 1, "stepsize_rule": "new"},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 50,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaSVRG", "max_epoch": 8, "R": 1},
    {"name": "AdaVRAE", "max_epoch": 50, "R": 1},
    {"name": "AdaVRAG", "max_epoch": 50, "R": 1},
    {"name": "SVRG", "max_epoch": 12, "R": 1, "H": 28.5},
    {"name": "AccSVRG", "max_epoch": 80, "R": 1, "H": 28.5},
]

EXP_GROUPS["logis_w8a"] = hu.cartesian_exp_group(
    {
        "dataset": "w8a",
        "model": ["logistic_regression"],
        "loss_func": ["logistic_loss"],
        "validation_metric": ["logistic_func_gap"],
        "acc_func": ["class2_acc"],
        "opt": opt_logis_w8a,
        "R": 1,
        "batch_size": 32,
        "sigma2": None,
        "record_grad_norm": True,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "runs": [0, 1, 2],
    }
)


# leu (L_max = 6720, solution can be inside the ball, d >> n)
opt_logis_leu = [
    {"name": "UniFastSgd", "max_epoch": 2000, "R": 1, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 2000,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSgd", "max_epoch": 4000, "R": 1, "stepsize_rule": "main"},
    {"name": "UniSgd-AdaGrad", "max_epoch": 4000, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniSvrg", "max_epoch": 15, "R": 1},
    {"name": "UniSvrg-AdaGrad", "max_epoch": 15, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniFastSvrg", "max_epoch": 1700, "R": 1},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 1700,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaSVRG", "max_epoch": 13, "R": 1},
    {"name": "AdaVRAE", "max_epoch": 1700, "R": 1},
    {"name": "AdaVRAG", "max_epoch": 1700, "R": 1},
    {"name": "SVRG", "max_epoch": 15, "R": 1, "H": 6720},
    {"name": "AccSVRG", "max_epoch": 2500, "R": 1, "H": 6720},
]

EXP_GROUPS["logis_leu"] = hu.cartesian_exp_group(
    {
        "dataset": "leu",
        "model": ["logistic_regression"],
        "loss_func": ["logistic_loss"],
        "validation_metric": ["logistic_func_gap"],
        "acc_func": ["class2_acc"],
        "opt": opt_logis_leu,
        "R": 1,
        "batch_size": 1,
        "sigma2": None,
        "record_grad_norm": True,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "runs": [0, 1, 2],
    }
)


# colon-cancer (L_max = 961.29, solution on the boundary, d >> n)
opt_logis_colon_cancer = [
    {"name": "UniFastSgd", "max_epoch": 2500, "R": 1, "stepsize_rule": "main"},
    {
        "name": "UniFastSgd-AdaGrad",
        "max_epoch": 2500,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "UniSgd", "max_epoch": 5000, "R": 1, "stepsize_rule": "main"},
    {"name": "UniSgd-AdaGrad", "max_epoch": 5000, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniSvrg", "max_epoch": 15, "R": 1},
    {"name": "UniSvrg-AdaGrad", "max_epoch": 15, "R": 1, "stepsize_rule": "AdaGrad"},
    {"name": "UniFastSvrg", "max_epoch": 2000, "R": 1},
    {
        "name": "UniFastSvrg-AdaGrad",
        "max_epoch": 2000,
        "R": 1,
        "stepsize_rule": "AdaGrad",
    },
    {"name": "AdaSVRG", "max_epoch": 13, "R": 1},
    {"name": "AdaVRAE", "max_epoch": 2000, "R": 1},
    {"name": "AdaVRAG", "max_epoch": 2000, "R": 1},
    {"name": "SVRG", "max_epoch": 15, "R": 1, "H": 961},
    {"name": "AccSVRG", "max_epoch": 2900, "R": 1, "H": 961},
]

EXP_GROUPS["logis_colon_cancer"] = hu.cartesian_exp_group(
    {
        "dataset": "colon-cancer",
        "model": ["logistic_regression"],
        "loss_func": ["logistic_loss"],
        "validation_metric": ["logistic_func_gap"],
        "acc_func": ["class2_acc"],
        "opt": opt_logis_colon_cancer,
        "R": 1,
        "batch_size": 1,
        "sigma2": None,
        "record_grad_norm": True,
        "replacement": True,
        "save_model": True,
        "use_cache_fstar": True,
        "runs": [0, 1, 2],
    }
)
