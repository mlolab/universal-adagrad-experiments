import copy

import numpy as np
import torch
import tqdm


class AccSVRG(torch.optim.Optimizer):
    """
    Implements the standard Accelerated SVRG Method with constant stepsize for convex optimization
                                                            (a primal version of the VRADA method).
    https://proceedings.neurips.cc/paper_files/paper/2020/file/093b60fd0557804c8ba0cbf1453da22f-Paper.pdf
    We only consider constaint optimization problems defined within a ball centered at the origin
    Therefore, the proximal mapping becomes the projection onto the ball.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        H (float): (1 / constant step size) for the standard AccSVRG (we assume H ~ L for this implementation)
        A0 (float): coefficient A_0 for the standard AccSVRG (A0 ~ 1 / steps_per_epoch)
        R (float): radius of the ball for projection. If None, then no projection is performed
        train_loader (torch.utils.data.dataloader.DataLoader): Dataloader for the training data
        model_base (torch.nn.Module): Base model for computing the full gradient
        loss_function (function handle): Loss function
        replacement (bool, optional): sampling with/without replacement
        steps_per_epoch (int, optional): number of steps per epoch.
            If set to None, it is m // 2 by default, where m is the number of minibatches in the training dataset
            If set to an int, it represents the customized number of steps per epoch.
    """

    def __init__(
        self,
        params,
        H,
        R,
        train_loader,
        model_base,
        loss_function,
        A0=None,
        replacement=True,
        steps_per_epoch=None,
    ):
        if not 0.0 < H:
            raise ValueError("Invalid H: {}".format(H))
        if not ((A0 is None) or (0.0 < A0)):
            raise ValueError("Invalid A0: {}".format(A0))
        if not ((R is None) or (0.0 < R)):
            raise ValueError("Invalid R: {}".format(R))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))

        # get m
        batch_size = train_loader.batch_size
        m = len(train_loader.dataset) // float(batch_size)
        if steps_per_epoch is None:
            steps_per_epoch = m // 2

        if not (1.0 <= steps_per_epoch):
            raise ValueError(
                "Invalid number of steps per epoch: {}. \
                perhaps the batchsize is larger than the number of training samples".format(
                    steps_per_epoch
                )
            )

        if A0 is None:
            A0 = 2 / m

        defaults = dict(
            H=H, R=R, replacement=replacement, steps_per_epoch=steps_per_epoch
        )

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["step"] = 0
        self.state["step_size"] = 1 / H
        self.state["A"] = A0
        self.state["nb_forwards"] = 0
        self.state["nb_backwards"] = 0
        self.state["nb_update_full_grad"] = 0

        for group in self.param_groups:
            params = group["params"]
            # get device
            for p in params:
                device = p.device
                self.state["device"] = device
                break
            # make sure that the initial params is inside the ball
            projection_ball(params, R)
            # we prove convergence for x_tilde which is stored in params
            self.state["x_tilde"] = copy.deepcopy(params)
            # at each epoch, Triangle SVRG Step starts at v
            self.state["v"] = copy.deepcopy(params)

    def compute_full_grad(self):
        """
        compute full gradient at params
        """

        device = self.state["device"]
        loader = torch.utils.data.DataLoader(
            self.state["train_loader"].dataset,
            drop_last=False,
            batch_size=self.state["train_loader"].batch_size,
        )
        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]
        model_base = self.state["model_base"]

        full_grad = []
        for batch in pbar:
            self.zero_grad()
            images, labels = (
                batch["images"].to(device=device),
                batch["labels"].to(device=device),
            )
            _ = loss_function(
                model_base, images, labels, aux=self.state["aux"], backwards=True
            )
            self.state["nb_forwards"] += 1
            self.state["nb_backwards"] += 1
            k = 0
            for p in model_base.parameters():
                if len(full_grad) == 0:
                    full_grad = [
                        torch.zeros_like(p.grad, requires_grad=False)
                        for p in model_base.parameters()
                    ]
                full_grad[k] += p.grad * images.shape[0]
                k += 1

        for g in full_grad:
            g /= len(self.state["train_loader"].dataset)

        return full_grad

    def _update_memory(self):
        """
        Update the full gradient at x_tilde
        """
        # set params to be x_tilde
        for group in self.param_groups:
            params = group["params"]
            set_params(params, self.state["x_tilde"])

        # compute the full gradient at x_tilde
        self.state["full_grad"] = self.compute_full_grad()

    def _proximal_step(self, grad_y, grad_tilde, step_size, R):
        """
        perform the proximal step
        """
        zipped = zip(
            self.state["v"],
            self.state["v"],
            grad_y,
            grad_tilde,
            self.state["full_grad"],
        )
        for v_next, v_current, g_y, g_tilde, g_full in zipped:
            v_next.data = v_current.data - step_size * (g_y - g_tilde + g_full)
        # projection to the ball
        projection_ball(self.state["v"], R)

    def compute_a(self):
        """
        compute a_{t+1} from the equation:
        a_{t+1}^2 / (A_t + a_{t+1}) + a_{t+1}^2 / A_t = 1
        """
        A = self.state["A"]
        p = np.poly1d([1, 2 * A, -1 * A, -A * A])
        a = np.max(p.roots)
        return a

    def step(self):
        """Performs one Triangle SVRG Epoch"""

        # record the initial step size
        if self.state.get("initial_step_size") is None:
            # we record 1 / H instead of a / H
            self.state["initial_step_size"] = self.state["step_size"]

        # get all the parameters
        for group in self.param_groups:
            steps_per_epoch = group["steps_per_epoch"]
            params = group["params"]
            H = group["H"]
            R = group["R"]
            replacement = group["replacement"]

        A = self.state["A"]
        device = self.state["device"]
        loader = torch.utils.data.DataLoader(
            self.state["train_loader"].dataset,
            drop_last=True,
            batch_size=self.state["train_loader"].batch_size,
            # sample with/without replacement
            sampler=torch.utils.data.sampler.RandomSampler(
                self.state["train_loader"].dataset,
                replacement=replacement,
                num_samples=int(
                    steps_per_epoch * self.state["train_loader"].batch_size
                ),
            ),
        )
        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]

        # compute a_{t+1}
        a = self.compute_a()

        # update the full gradient at x_tilde
        self._update_memory()
        self.state["nb_update_full_grad"] += 1

        # set xbar to be zeros
        xbar = copy.deepcopy(params)
        for p in xbar:
            p.data = torch.zeros_like(p.data, requires_grad=False)

        k = 0

        # start steps_per_epoch iterations of Triangle SVRG Steps
        for batch in pbar:
            # set params to be y
            linear_combination(
                params, A / (A + a), self.state["x_tilde"], a / (A + a), self.state["v"]
            )

            self.zero_grad()
            images, labels = (
                batch["images"].to(device=device),
                batch["labels"].to(device=device),
            )

            # compute minibatch gradient at y
            _ = loss_function(
                self.state["model_base"],
                images,
                labels,
                aux=self.state["aux"],
                backwards=True,
            )
            grad_y = get_grad_list(params)
            self.state["nb_forwards"] += 1
            self.state["nb_backwards"] += 1

            # compute minibatch gradient at x_tilde
            self.zero_grad()
            set_params(params, self.state["x_tilde"])
            _ = loss_function(
                self.state["model_base"],
                images,
                labels,
                aux=self.state["aux"],
                backwards=True,
            )
            grad_tilde = get_grad_list(params)
            self.state["nb_forwards"] += 1
            self.state["nb_backwards"] += 1

            # update v+
            self._proximal_step(grad_y, grad_tilde, a / H, R)

            # set params to be x+
            linear_combination(
                params, A / (A + a), self.state["x_tilde"], a / (A + a), self.state["v"]
            )

            # update xbar
            zipped = zip(xbar, xbar, params)
            with torch.no_grad():
                for xbar_next, xbar_current, x_next in zipped:
                    xbar_next.data = (1 - 1 / (k + 1)) * xbar_current.data + (
                        1 / (k + 1)
                    ) * x_next.data

            k += 1
            if k >= steps_per_epoch:
                # update x_tilde
                self.state["x_tilde"] = xbar

                # set params to x_tilde for evaluation
                set_params(params, self.state["x_tilde"])
                break

        # udpate A_{t+1}
        self.state["A"] += a
        self.state["step"] += 1
        return None


# helper functions
def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]


def set_params(params, params_next):
    for p, p_next in zip(params, params_next):
        p.data = p_next.data


def linear_combination(y, a, params_1, b, params_2):
    for y_next, p1, p2 in zip(y, params_1, params_2):
        y_next.data = a * p1.data + b * p2.data


def projection_ball(params, R):
    if R is not None:
        # compute norm
        p_norm2 = torch.tensor([0]).double()
        for p in params:
            p_norm2 += torch.sum(torch.mul(p.data, p.data))
        p_norm = torch.sqrt(p_norm2)
        # projection to the ball
        if p_norm > R:
            for p in params:
                p.data = p.data / p_norm * R
    else:
        pass
