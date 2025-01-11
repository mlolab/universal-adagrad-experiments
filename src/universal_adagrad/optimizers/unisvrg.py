import copy

import numpy as np
import torch
import tqdm


class UniSVRG(torch.optim.Optimizer):
    """
    Implements UniSvrg (Alg.3) for convex optimization.
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
        replacement (bool, optional): sampling with/without replacement
            for the current implementation, we only consider sampling with
                replacement for the new and AdaGrad stepsize rules
        stepsize_rule (string optinal): 'new', 'old' or 'AdaGrad'
        steps_per_epoch (int, optional):
            If set to None, it is 2^k by default, where k is epoch number (starting from 0)
            If set to an int, it represents the constant number of steps per epoch.
    """

    def __init__(
        self,
        params,
        R,
        train_loader,
        model_base,
        loss_function,
        replacement=True,
        stepsize_rule="new",
        steps_per_epoch=None,
    ):
        if not (0.0 < R):
            raise ValueError("Invalid R: {}".format(R))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))
        if stepsize_rule not in ["new", "old", "AdaGrad"]:
            raise ValueError("Invalid stepsize_rule: {}".format(stepsize_rule))
        if not ((steps_per_epoch is None) or (1.0 <= steps_per_epoch)):
            raise ValueError(
                "Invalid number of steps per epoch: {}".format(steps_per_epoch)
            )

        defaults = dict(R=R, replacement=replacement, stepsize_rule=stepsize_rule)

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["step"] = 0
        # stepsize is defined by 1 / H_t where H_0 is 0
        self.state["step_size"] = np.inf
        self.state["nb_forwards"] = 0
        self.state["nb_backwards"] = 0
        self.state["nb_update_full_grad"] = 0
        if steps_per_epoch is None:
            self.state["steps_per_epoch"] = 2
            self.state["increase_steps_per_epoch"] = True
        else:
            self.state["steps_per_epoch"] = steps_per_epoch
            self.state["increase_steps_per_epoch"] = False

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
            # at each epoch, SvrgStep starts at x
            self.state["x"] = copy.deepcopy(params)

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
                model_base, images, labels, self.state["aux"], backwards=True
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

    def update_stepsize(self, params_current, grad_current, grad_tilde, stepsize_rule):
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
        xplus = copy.deepcopy(params)

        # generate an independent minibatch sample
        for batch in pbar:
            self.zero_grad()
            images, labels = (
                batch["images"].to(device=device),
                batch["labels"].to(device=device),
            )
            # compute stochastic loss value at params (xplus)
            loss_xplus_prime = loss_function(
                self.state["model_base"],
                images,
                labels,
                aux=self.state["aux"],
                backwards=True,
            )
            self.state["nb_forwards"] += 1
            if stepsize_rule in ["new", "AdaGrad"]:
                grad_xplus_prime = get_grad_list(params)
                self.state["nb_backwards"] += 1

            if stepsize_rule in ["old"]:
                # compute loss value and minibatch gradient at params_current (x)
                self.zero_grad()
                set_params(params, params_current)
                loss_current_prime = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    self.state["aux"],
                    backwards=True,
                )
                grad_current_prime = get_grad_list(params)
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

            # compute minibatch gradient at x_tilde
            self.zero_grad()
            set_params(params, self.state["x_tilde"])
            _ = loss_function(
                self.state["model_base"],
                images,
                labels,
                self.state["aux"],
                backwards=True,
            )
            grad_tilde_prime = get_grad_list(params)
            self.state["nb_forwards"] += 1
            self.state["nb_backwards"] += 1

            break

        if stepsize_rule in ["old"]:
            # compute brgeman distance
            breg_dist = compute_bregman_distance(
                loss_xplus_prime,
                loss_current_prime,
                grad_current_prime,
                params_current,
                xplus,
            )

            # compute <G_1 - G, xplus - x>
            diff_inner_prod = 0.0
            for g, g_tilde, g_prime, g_tilde_prime, pplus, p in zip(
                grad_current,
                grad_tilde,
                grad_current_prime,
                grad_tilde_prime,
                xplus,
                params_current,
            ):
                diff_inner_prod += torch.sum(
                    torch.mul(
                        pplus.data - p.data, g_prime - g_tilde_prime - g + g_tilde
                    )
                )

        elif stepsize_rule in ["new"]:
            # compute <G+ - G, xplus - x>
            diff_inner_prod = 0.0
            for g, g_tilde, g_plus_prime, g_tilde_prime, pplus, p in zip(
                grad_current,
                grad_tilde,
                grad_xplus_prime,
                grad_tilde_prime,
                xplus,
                params_current,
            ):
                diff_inner_prod += torch.sum(
                    torch.mul(
                        pplus.data - p.data, g_plus_prime - g_tilde_prime - g + g_tilde
                    )
                )

        if stepsize_rule in ["new", "old"]:
            # compute squared distance between prams_current and xplus
            squared_dist = 0.0
            for pplus, p in zip(xplus, params_current):
                squared_dist += torch.sum(
                    torch.mul(pplus.data - p.data, pplus.data - p.data)
                )

        if stepsize_rule in ["AdaGrad"]:
            # compute squared distance between G+ and G
            squared_dist = 0.0
            for g, gtilde, g_plus_prime, g_tilde_prime in zip(
                grad_current, grad_tilde, grad_xplus_prime, grad_tilde_prime
            ):
                squared_dist += torch.sum(
                    torch.mul(
                        g_plus_prime - g_tilde_prime - g + gtilde,
                        g_plus_prime - g_tilde_prime - g + gtilde,
                    )
                )

        # compute H+
        # tentative stepsize
        if stepsize_rule in ["old"]:
            Hplus = (
                (H * (D**2) + breg_dist + diff_inner_prod) / ((D**2) + squared_dist / 2)
            ).item()
            # check
            if breg_dist + diff_inner_prod - Hplus / 2 * squared_dist < 0:
                Hplus = H

        elif stepsize_rule in ["new"]:
            Hplus = (
                (H * (D**2) + diff_inner_prod) / ((D**2) + squared_dist / 2)
            ).item()

            # check
            if diff_inner_prod - Hplus / 2 * squared_dist < 0:
                Hplus = H

        elif stepsize_rule in ["AdaGrad"]:
            Hplus = np.sqrt(H**2 + squared_dist.item() / (D**2))

        if Hplus == 0.0:
            self.state["step_size"] = np.inf
        else:
            self.state["step_size"] = 1 / Hplus

        # set params back to xplus
        set_params(params, xplus)

        if stepsize_rule in ["new", "AdaGrad"]:
            return grad_xplus_prime, grad_tilde_prime

    def step(self):
        """Performs one SVRG epoch."""

        # record the initial step size
        if self.state.get("initial_step_size") is None:
            self.state["initial_step_size"] = self.state["step_size"]

        # get all the parameters
        for group in self.param_groups:
            params = group["params"]
            R = group["R"]
            stepsize_rule = group["stepsize_rule"]
            replacement = group["replacement"]

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
                    self.state["steps_per_epoch"]
                    * self.state["train_loader"].batch_size
                ),
            ),
        )

        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]

        # update the full gradient at x_tilde
        self._update_memory()
        self.state["nb_update_full_grad"] += 1

        # set params to be x
        set_params(params, self.state["x"])
        # set xbar to be x
        xbar = copy.deepcopy(self.state["x"])

        k = 0

        # start steps_per_epoch iterations of UniSvrgSteps
        for batch in pbar:
            params_current = copy.deepcopy(params)
            self.zero_grad()
            if (k == 0 and stepsize_rule in ["new", "AdaGrad"]) or (
                stepsize_rule in ["old"]
            ):
                images, labels = (
                    batch["images"].to(device=device),
                    batch["labels"].to(device=device),
                )

                # compute minibatch gradient at x
                _ = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    aux=self.state["aux"],
                    backwards=True,
                )
                grad_current = get_grad_list(params)
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

                # compute minibatch gradient at x_tilde
                self.zero_grad()
                set_params(params, self.state["x_tilde"])
                _ = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    self.state["aux"],
                    backwards=True,
                )
                grad_tilde = get_grad_list(params)
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

            # update x
            proximal_step(
                params,
                params_current,
                grad_current,
                grad_tilde,
                self.state["full_grad"],
                self.state["step_size"],
                R,
            )

            # update stepsize
            if stepsize_rule in ["new", "AdaGrad"]:
                # update the next grad_current and grad_tilde
                grad_current, grad_tilde = self.update_stepsize(
                    params_current, grad_current, grad_tilde, stepsize_rule
                )
            else:
                self.update_stepsize(
                    params_current, grad_current, grad_tilde, stepsize_rule
                )

            # update xbar
            zipped = zip(xbar, xbar, params)
            with torch.no_grad():
                for xbar_next, xbar_current, x_next in zipped:
                    xbar_next.data = (1 - 1 / (k + 1)) * xbar_current.data + (
                        1 / (k + 1)
                    ) * x_next.data

            k += 1
            if k >= self.state["steps_per_epoch"]:
                # update x_tilde and x
                self.state["x_tilde"] = xbar
                self.state["x"] = copy.deepcopy(params)

                # set params to x_tilde for evaluation
                set_params(params, self.state["x_tilde"])
                break

        self.state["step"] += 1
        if self.state["increase_steps_per_epoch"]:
            self.state["steps_per_epoch"] = 2 ** (self.state["step"] + 1)

        return None


# helper functions
def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]


def set_params(params, params_next):
    for p, p_next in zip(params, params_next):
        p.data = p_next.data


def compute_bregman_distance(loss_xplus, loss_x, grad_x, x, xplus):
    # return loss_xplus - loss_x - <grad_x, xplus - x>
    with torch.no_grad():
        innerprod = 0.0
        for p, pplus, g in zip(x, xplus, grad_x):
            innerprod += torch.sum(torch.mul(pplus.data - p.data, g))

    return loss_xplus - loss_x - innerprod


def proximal_step(
    params, params_current, grad_current, grad_tilde, full_grad_tilde, step_size, R
):
    with torch.no_grad():
        if step_size == np.inf:
            step_size = 1e9
        zipped = zip(params, params_current, grad_current, grad_tilde, full_grad_tilde)
        for x_next, x_current, g_current, g_tilde, g_full in zipped:
            x_next.data = x_current.data - step_size * (g_current - g_tilde + g_full)
        # projection to the ball
        projection_ball(params, R)


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
