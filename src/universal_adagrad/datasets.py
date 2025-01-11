import os

import numpy as np
import pooch
import torch
import torchvision
from scipy.optimize import NonlinearConstraint, minimize
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from torchvision import transforms


def get_dataset(dataset_name, split, datadir, exp_dict):
    train_flag = True if split == "train" else False

    if dataset_name in ["polyhedron_dataset"]:
        # A has shape (nb_samples, dim) that stores a_i^T on each row
        # b has shape (nb_samples,)
        # Ax^* <= b
        n = exp_dict["nb_samples"]
        d = exp_dict["d"]
        R = exp_dict["R"]

        rng = np.random.default_rng(2024)
        xstar = rng.normal(size=(d), loc=0) * 100
        xstar = xstar / np.linalg.norm(xstar) * 0.95 * R

        A = rng.random((n, d)) * 2 - 1
        Axstar = A @ xstar
        s = np.where(Axstar < 0, Axstar, -np.inf).argmax()
        S = rng.random((n)) * s / 10

        b = A @ xstar + S
        dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(A), torch.DoubleTensor(b)
        )
        fstar = 0.0
        aux = {"fstar": fstar, "R": R, "q": exp_dict["q"], "xstar": xstar}

        if train_flag:
            print("------------------fstar={:.2f}-------------------".format(fstar))

    if dataset_name in ["L2_syn_dataset"]:
        # A has shape (nb_samples, dim)
        # which stores the diagonal matrices {A_i}
        # b has shape (nb_samples, dim)
        # which stores the vectors {b_i}
        n = exp_dict["nb_samples"]
        d = exp_dict["d"]
        rng_A = np.random.default_rng(2023)
        rng_b = np.random.default_rng(2024)
        A = np.clip(np.abs((rng_A.random((n, d)) * 100)), a_min=1, a_max=100)
        b = rng_b.random((n, d)) * 10

        dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(A), torch.DoubleTensor(b)
        )
        if train_flag:
            #  save some time by using pre-computed fstar
            if exp_dict["use_cache_fstar"]:
                if n == 100 and d == 1000 and exp_dict["R"] == 3:
                    fstar = 120.13651170621044
                    solution_norm = 2.3789
                elif n == 100 and d == 1000 and exp_dict["R"] == 1:
                    fstar = 143.87112011829703
                    solution_norm = 1.0
                elif n == 1000 and d == 100 and exp_dict["R"] == 1:
                    fstar = 38.058440666829426
                    solution_norm = 0.7482801277303296
                elif n == 1000 and d == 100 and exp_dict["R"] == 0.5:
                    fstar = 40.65683254453336
                    solution_norm = 0.5
                else:
                    print("No cache is found.")
                    print("------------------compute fstar-------------------")
                    fstar, solution_norm = compute_fstar(
                        torch.DoubleTensor(A), torch.DoubleTensor(b), exp_dict
                    )
            else:
                # compute fstar for the training set
                print("------------------compute fstar-------------------")
                fstar, solution_norm = compute_fstar(
                    torch.DoubleTensor(A), torch.DoubleTensor(b), exp_dict
                )
            print("------------------fstar={:.2f}-------------------".format(fstar))
            print(
                "------------------solution_norm={:.2f}-------------------".format(
                    solution_norm
                )
            )

            aux = {"fstar": fstar, "solution_norm": solution_norm}
        else:
            aux = {}

    if dataset_name in [
        "mushrooms",
        "a1a",
        "a2a",
        "ijcnn",
        "w8a",
        "breast-cancer",
        "duke",
        "leu",
        "phishing",
        "rcv1",
        "epsilon",
        "colon-cancer",
    ]:
        X, y = load_libsvm(dataset_name, data_dir=datadir)

        labels = np.unique(y)

        y[y == labels[0]] = -1
        y[y == labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=9513451
        )
        X_train, X_test, Y_train, Y_test = splits

        if train_flag:
            # training set
            X_train = torch.DoubleTensor(X.toarray())
            Y_train = torch.DoubleTensor(y)
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            dataset.targets = Y_train.tolist()

            # compute L for the training set
            L = compute_L(X_train, Y_train, exp_dict)
            if L is not None:
                print("------------------L={:.2f}--------------------".format(L))

            #  save some time by using pre-computed fstar
            if exp_dict["use_cache_fstar"] and (exp_dict["R"] == 1):
                if (dataset_name == "w8a") and (
                    exp_dict["validation_metric"] == "logistic_func_gap"
                ):
                    fstar = 0.3694144128280212
                    solution_norm = 1.0
                elif (dataset_name == "mushrooms") and (
                    exp_dict["validation_metric"] == "logistic_func_gap"
                ):
                    fstar = 0.3208744580259251
                    solution_norm = 1.0
                elif (dataset_name == "leu") and (
                    exp_dict["validation_metric"] == "logistic_func_gap"
                ):
                    fstar = 0.0
                    solution_norm = 0.9948
                elif (dataset_name == "colon-cancer") and (
                    exp_dict["validation_metric"] == "logistic_func_gap"
                ):
                    fstar = 0.010186023850834041
                    solution_norm = 1.0
                elif (dataset_name == "mushrooms") and (
                    exp_dict["validation_metric"] == "hinge_func_gap"
                ):
                    fstar = 0.13838872506568312
                    solution_norm = 1.0
                elif (dataset_name == "w8a") and (
                    exp_dict["validation_metric"] == "hinge_func_gap"
                ):
                    fstar = 0.33394709335312434
                    solution_norm = 1.0
                elif (dataset_name == "leu") and (
                    exp_dict["validation_metric"] == "hinge_func_gap"
                ):
                    fstar = -1e-20
                    solution_norm = 0.6649
                elif (dataset_name == "colon-cancer") and (
                    exp_dict["validation_metric"] == "hinge_func_gap"
                ):
                    fstar = -1e-20
                    solution_norm = 0.7643
                elif (
                    (dataset_name == "mushrooms")
                    and (exp_dict["validation_metric"] == "huber_func_gap")
                    and (exp_dict.get("mu_huber") == 1e-3)
                ):
                    fstar = 0.15586571151327158
                    solution_norm = 1.0
                elif (
                    (dataset_name == "w8a")
                    and (exp_dict["validation_metric"] == "huber_func_gap")
                    and (exp_dict.get("mu_huber") == 1e-3)
                ):
                    fstar = 0.5129655714416271
                    solution_norm = 1.0
                elif (
                    (dataset_name == "leu")
                    and (exp_dict["validation_metric"] == "huber_func_gap")
                    and (exp_dict.get("mu_huber") == 1e-3)
                ):
                    fstar = 0.0
                    solution_norm = 0.0721
                elif (
                    (dataset_name == "colon-cancer")
                    and (exp_dict["validation_metric"] == "huber_func_gap")
                    and (exp_dict.get("mu_huber") == 1e-3)
                ):
                    fstar = 0.2898226046248009
                    solution_norm = 0.2121
                else:
                    print("No cache is found.")
                    # compute fstar for the training set
                    print("------------------compute fstar-------------------")
                    fstar, solution_norm = compute_fstar(X_train, Y_train, exp_dict)
            else:
                # compute fstar for the training set
                print("------------------compute fstar-------------------")
                fstar, solution_norm = compute_fstar(X_train, Y_train, exp_dict)
            print("------------------fstar={:.2f}-------------------".format(fstar))
            print(
                "------------------solution_norm={:.2f}-------------------".format(
                    solution_norm
                )
            )

            aux = {"fstar": fstar, "L": L}
            if exp_dict.get("mu_huber") is not None:
                aux["mu_huber"] = exp_dict["mu_huber"]

        else:
            # test set
            X_test = torch.DoubleTensor(X_test.toarray())
            Y_test = torch.DoubleTensor(Y_test)
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
            dataset.targets = Y_test.tolist()
            aux = {}

    if dataset_name == "cifar10":
        transform_function_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_function_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=False,
            transform=transform_function_train
            if train_flag
            else transform_function_test,
        )

        aux = {}

    if dataset_name == "cifar100":
        transform_function_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_function_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function_train
            if train_flag
            else transform_function_test,
        )

        aux = {}

    # get the coefficient for L2 regularizer
    aux["sigma2"] = exp_dict.get("sigma2", None)
    dataset.aux = aux

    return DatasetWrapper(dataset, split=split, aux=aux)


class DatasetWrapper:
    def __init__(self, dataset, split, aux):
        self.dataset = dataset
        self.split = split
        self.aux = aux

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return {"images": data, "labels": target, "meta": {"indices": index}}


# ===========================================================
# Helpers
LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {
    "mushrooms": "mushrooms",
    "a1a": "a1a",
    "a2a": "a2a",
    "ijcnn": "ijcnn1.tr.bz2",
    "w8a": "w8a",
    "breast-cancer": "breast-cancer",
    "duke": "duke.tr.bz2",
    "leu": "leu.bz2",
    "phishing": "phishing",
    "rcv1": "rcv1_train.binary.bz2",
    "epsilon": "epsilon_normalized.bz2",
    "colon-cancer": "colon-cancer.bz2",
}


def load_libsvm(name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    fn = LIBSVM_DOWNLOAD_FN[name]

    # url = urllib.parse.urljoin(LIBSVM_URL, fn)
    downloader = pooch.create(
        path=data_dir,
        base_url=LIBSVM_URL,
        registry={
            "mushrooms": "f39a4eb628dc61a7d43760815b061c9e497aa728ce1ad8bde57a09ef6043b538",
            "w8a": "6a9fa8fd5f524303240a5db07d4b3d4a51e8b7b4b20a914105d8e3e8c81640f2",
            "leu.bz2": "2f7dc1c8ea4343b17038e27ff777dad0a87d79a5fa29de54765055cc841cb75b",
            "colon-cancer.bz2": "724d7291efb9a833b8407442a27f70f64beeb86c107d7b3ce98e8c48fe878867",
        },
    )
    # Fetch the file (download if not cached)
    try:
        file_path = downloader.fetch(fn)
        print(f"Data file is cached at: {file_path}")
    except Exception as e:
        print(f"Error downloading {fn}: {e}")
        raise

    X, y = load_svmlight_file(file_path)
    return X, y


def compute_L(X_train, Y_train, exp_dict):
    """
    only implements L_{max} now
    todo: implement a tighter bound
    """
    # X_train_np is of shape (nb_samples, dim)
    # Y_train consists 1 and -1, which does not affect L
    X_train_np = X_train.numpy()

    if exp_dict["validation_metric"] == "logistic_func_gap":
        # L <= 1/4 * ||X_train||^2
        L = (X_train_np * X_train_np).sum(axis=1).max() / 4
    else:
        L = None

    return L


def compute_fstar(X_train, Y_train, exp_dict):
    R = exp_dict["R"]
    if R is not None:
        cons = NonlinearConstraint(lambda x: x @ x - R**2, -np.inf, 0.0)

    # X_train_np is of shape (nb_samples, dim)
    X_train_np = X_train.numpy()
    # Y_train_np stores labels (-1 or 1) of shape (nb_samples,)
    Y_train_np = Y_train.numpy()

    sigma2 = 0.0 if exp_dict.get("sigma2") is None else exp_dict["sigma2"]

    if exp_dict["validation_metric"] == "logistic_func_gap":
        # f_i(x) = ln (1 + exp(-Y_train_i * (X_train_i^T x_i)))
        def func(x):
            return np.log(
                1 + np.exp(-1 * Y_train_np * (X_train_np @ x))
            ).mean() + sigma2 / 2 * (x @ x)

    elif exp_dict["validation_metric"] == "hinge_func_gap":
        # f_i(x) = [1 - Y_train_i * (X_train_i^T x)]_+
        def func(x):
            return np.maximum(
                0, 1 - Y_train_np * (X_train_np @ x)
            ).mean() + sigma2 / 2 * (x @ x)

    elif exp_dict["validation_metric"] == "huber_func_gap":
        # shifted hyber loss
        mu = exp_dict["mu_huber"]

        def func(x):
            return np.where(
                np.abs(1 - Y_train_np * (X_train_np @ x)) < mu,
                (1 - Y_train_np * (X_train_np @ x)) ** 2 / (2 * mu),
                np.abs(1 - Y_train_np * (X_train_np @ x)) - mu / 2,
            ).mean() + sigma2 / 2 * (x @ x)

    elif exp_dict["validation_metric"] == "L2_func_gap":
        # f_i(x) = ||X_train_i @ x - Y_train_i||_2 where X_train_i is a diagonal matrix stored in a row
        # Y_train_np stores vectors of shape (nb_samples, dim)
        def func(x):
            return np.sqrt(
                ((X_train_np * x.reshape(1, len(x)) - Y_train_np) ** 2).sum(axis=1)
            ).mean()
    else:
        # not implemented yet for other metrics
        return None, None

    if R is not None:
        res = minimize(
            func,
            np.random.rand(X_train_np.shape[1]) * 1e-2,
            constraints=cons,
            tol=10**-16,
        )
    else:
        res = minimize(func, np.random.rand(X_train_np.shape[1]) * 1e-2, tol=10**-16)
    solution_norm = np.linalg.norm(res.x)

    return res.fun, solution_norm
