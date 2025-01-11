import copy

import numpy as np
import torch
import tqdm


class UniFastSGD(torch.optim.Optimizer):
    """
    Implements UniFastSgd (Alg.2) for convex optimization.
    In this implementation, we only consider constaint optimization problems
    defined within a ball centered at the origin
    Therefore, the proximal mapping becomes the projection onto the ball.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        R (float): radius of the ball for projection. If None, no projection is performed
        train_loader (torch.utils.data.dataloader.DataLoader): Dataloader for the training data
        model_base (torch.nn.Module): Base model for computing the full gradient
        loss_function (function handle): Loss function
        stepsize_rule (string optinal): 'main' or 'AdaGrad'
        steps_per_epoch (int, optional):
            If set to an int, it represents the constant number of steps per epoch.
            By default, it is set to (total number of data points / batch size)
    """

    def __init__(
        self,
        params,
        R,
        train_loader,
        model_base,
        loss_function,
        stepsize_rule="main",
        steps_per_epoch=None,
    ):
        if not (0.0 < R):
            raise ValueError("Invalid R: {}".format(R))
        if stepsize_rule not in ["main", "AdaGrad"]:
            raise ValueError("Invalid stepsize_rule: {}".format(stepsize_rule))
        if not ((steps_per_epoch is None) or (1.0 <= steps_per_epoch)):
            raise ValueError(
                "Invalid number of steps per epoch: {}".format(steps_per_epoch)
            )

        defaults = dict(R=R, stepsize_rule=stepsize_rule)

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["step"] = 0
        # stepsize is defined by 1 / H_t where H_0 is 0
        self.state["step_size"] = np.inf
        self.state["A"] = 0
        self.state["nb_forwards"] = 0
        self.state["nb_backwards"] = 0
        if steps_per_epoch is None:
            batch_size = train_loader.batch_size
            nb_data = len(train_loader.dataset)
            self.state["steps_per_epoch"] = nb_data // batch_size
        else:
            self.state["steps_per_epoch"] = steps_per_epoch

        for group in self.param_groups:
            params = group["params"]
            # get device
            for p in params:
                device = p.device
                self.state["device"] = device
                break
            # make sure that the initial params is inside the ball
            projection_ball(params, R)
            # at each epoch, Triangle SVRG Step starts at v
            self.state["v"] = copy.deepcopy(params)

    def update_stepsize(self, a, y_current, grad_y, stepsize_rule):
        """
        update H+
        """
        # get all the parameters
        for group in self.param_groups:
            params = group["params"]
            R = group["R"]
        D = 2 * R
        H = 1 / self.state["step_size"]
        device = self.state["device"]
        loader = torch.utils.data.DataLoader(
            self.state["train_loader"].dataset,
            drop_last=True,
            batch_size=self.state["train_loader"].batch_size,
            shuffle=True,
        )
        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]
        Aplus = self.state["A"] + a

        # generate an independent minibatch sampled at the next point
        for batch in pbar:
            self.zero_grad()
            images, labels = (
                batch["images"].to(device=device),
                batch["labels"].to(device=device),
            )
            # compute stochastic loss value at params (xplus)
            _ = loss_function(
                self.state["model_base"],
                images,
                labels,
                aux=self.state["aux"],
                backwards=True,
            )
            grad_xplus_prime = get_grad_list(params)
            self.state["nb_forwards"] += 1
            self.state["nb_backwards"] += 1
            break

        if stepsize_rule in ["main"]:
            # compute <g_x+ - g_y, xplus - y>
            diff_inner_prod = 0.0
            for g_y, g_plus_prime, pplus, y in zip(
                grad_y, grad_xplus_prime, params, y_current
            ):
                diff_inner_prod += torch.sum(
                    torch.mul(pplus.data - y.data, g_plus_prime - g_y)
                )

            # compute ||x+ - y||^2
            squared_dist = 0.0
            for pplus, y in zip(params, y_current):
                squared_dist += torch.sum(
                    torch.mul(pplus.data - y.data, pplus.data - y.data)
                )

            Hplus = (
                ((H * (D**2) / Aplus) + diff_inner_prod)
                / ((a**2) * (D**2) / (Aplus**2) + squared_dist / 2)
            ).item()

            # check
            if diff_inner_prod - Hplus / 2 * squared_dist < 0:
                Hplus = H
            else:
                Hplus = (a**2) / Aplus * Hplus

        else:
            # compute ||g_xplus - g_y||^2
            squared_dist = 0.0
            for g_y, g_plus_prime in zip(grad_y, grad_xplus_prime):
                squared_dist += torch.sum(
                    torch.mul(g_plus_prime - g_y, g_plus_prime - g_y)
                )

            Hplus = np.sqrt(H**2 + (a**2) / (D**2) * squared_dist.item())

        if Hplus == 0.0:
            self.state["step_size"] = np.inf
        else:
            self.state["step_size"] = 1 / Hplus

    def step(self):
        """Performs one epoch of UniTriSteps"""

        # record the initial step size
        if self.state.get("initial_step_size") is None:
            self.state["initial_step_size"] = self.state["step_size"]

        # get all the parameters
        for group in self.param_groups:
            params = group["params"]
            R = group["R"]
            stepsize_rule = group["stepsize_rule"]

        device = self.state["device"]
        loader = torch.utils.data.DataLoader(
            self.state["train_loader"].dataset,
            drop_last=True,
            batch_size=self.state["train_loader"].batch_size,
            # sample with/without replacement
            sampler=torch.utils.data.sampler.RandomSampler(
                self.state["train_loader"].dataset,
                replacement=True,
                num_samples=int(
                    self.state["steps_per_epoch"]
                    * self.state["train_loader"].batch_size
                ),
            ),
        )

        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]

        j = 0

        # start steps_per_epoch iterations of UniTriSteps
        for batch in pbar:
            k = self.state["step"] * self.state["steps_per_epoch"] + j
            # update a
            a = (k + 1) / 2
            A = self.state["A"]

            params_current = copy.deepcopy(params)
            # set params to be y
            linear_combination(
                params, A / (A + a), params_current, a / (A + a), self.state["v"]
            )
            y_current = copy.deepcopy(params)

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

            # update v
            proximal_step(
                self.state["v"], self.state["v"], grad_y, a * self.state["step_size"], R
            )

            # update x
            linear_combination(
                params, A / (A + a), params_current, a / (A + a), self.state["v"]
            )

            # update stepsize
            self.update_stepsize(a, y_current, grad_y, stepsize_rule)

            # update A
            self.state["A"] += a

            j += 1
            if j >= self.state["steps_per_epoch"]:
                break

        self.state["step"] += 1

        return None


# helper functions
def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]


def set_params(params, params_next):
    for p, p_next in zip(params, params_next):
        p.data = p_next.data


def proximal_step(params, params_current, grad_current, step_size, R):
    with torch.no_grad():
        if step_size == np.inf:
            step_size = 1e9
        zipped = zip(params, params_current, grad_current)
        for (
            x_next,
            x_current,
            g_current,
        ) in zipped:
            x_next.data = x_current.data - step_size * g_current
        # projection to the ball
        projection_ball(params, R)


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
