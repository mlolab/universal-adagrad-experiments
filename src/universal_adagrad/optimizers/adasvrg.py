import copy

import numpy as np
import torch
import tqdm


class AdaSVRG(torch.optim.Optimizer):
    """
    Implements the (Multi-stage) AdaSVRG for convex optimization.
    https://arxiv.org/pdf/2102.09645.pdf (Algorithm 2)
    In this implementation, we only consider constaint optimization problems
    defined within a ball centered at the origin

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        R (float): radius of the ball for projection. If None, no projection is performed
        K (int): number of outer loops
        eta (float): eta parameter (according to the theory, the best eta is ~D which is 2R in this case)
        train_loader (torch.utils.data.dataloader.DataLoader): Dataloader for the training data
        model_base (torch.nn.Module): Base model for computing the full gradient
        loss_function (function handle): Loss function
        replacement (bool, optional): sampling with/without replacement
        steps_per_epoch (int, optional):
            If set to None, it is 2^{i+1} by default, where i is the stage index (starting from 1)
            If set to an int, it represents the customized number of steps per epoch.
    """

    def __init__(
        self,
        params,
        R,
        train_loader,
        model_base,
        loss_function,
        K=3,
        eta=None,
        replacement=True,
        steps_per_epoch=None,
    ):
        if not (0.0 < R):
            raise ValueError("Invalid R: {}".format(R))
        if not (1 <= K):
            raise ValueError("Invalid K: {}".format(K))
        if not ((eta is None) or (0.0 < eta)):
            raise ValueError("Invalid eta: {}".format(eta))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))
        if not ((steps_per_epoch is None) or (1.0 <= steps_per_epoch)):
            raise ValueError(
                "Invalid number of steps per epoch: {}".format(steps_per_epoch)
            )

        if eta is None:
            eta = 2 * R

        defaults = dict(R=R, K=K, eta=eta, replacement=replacement)

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["A"] = 0.0
        self.state["step"] = 0
        # stepsize is defined by eta / A_t
        self.state["step_size"] = 0.0
        self.state["nb_forwards"] = 0
        self.state["nb_backwards"] = 0
        self.state["nb_update_full_grad"] = 0
        if steps_per_epoch is None:
            self.state["steps_per_epoch"] = 2**2
            self.state["increase_steps_per_epoch"] = True
        else:
            self.state["steps_per_epoch"] = steps_per_epoch
            self.state["increase_steps_per_epoch"] = False

        # we prove convergence for w_bar which is stored in params
        for group in self.param_groups:
            params = group["params"]
            # get device
            for p in params:
                device = p.device
                self.state["device"] = device
                break
            # make sure that the initial params is inside the ball
            projection_ball(params, R)
            # prove convergence for w_bar
            self.state["w_bar"] = copy.deepcopy(params)
            # full gradient is updated at w
            self.state["w"] = copy.deepcopy(params)

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
        Update the full gradient at w_k
        """
        # set params to be w_k
        for group in self.param_groups:
            params = group["params"]
            set_params(params, self.state["w"])

        # compute the full gradient at w_k
        self.state["full_grad"] = self.compute_full_grad()

    def _update_accumulator(self, grad_current, grad_tilde, t):
        """
        Update the accumulator A_t
        """
        if t == 0:
            self.state["A"] = 0.0
        gnorm2 = torch.tensor([0]).double()
        for g, g_tilde, g_full in zip(
            grad_current, grad_tilde, self.state["full_grad"]
        ):
            gnorm2 += torch.sum(torch.mul(g - g_tilde + g_full, g - g_tilde + g_full))
        self.state["A"] = np.sqrt(self.state["A"] ** 2 + gnorm2.item())

    def step(self):
        """Performs one stage of AdaSVRG."""

        # get all the parameters
        for group in self.param_groups:
            params = group["params"]
            R = group["R"]
            K = group["K"]
            eta = group["eta"]
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

        # start K outer loops
        for k in range(K):
            w_record = copy.deepcopy(self.state["w"])
            # update the full gradient at w_k
            self._update_memory()
            self.state["nb_update_full_grad"] += 1

            # set params to be w_k
            set_params(params, self.state["w"])

            # start steps_per_epoch inner loops
            t = 0

            for batch in pbar:
                params_current = copy.deepcopy(params)
                self.zero_grad()
                images, labels = (
                    batch["images"].to(device=device),
                    batch["labels"].to(device=device),
                )

                # compute minibatch gradient at x_t
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

                # compute minibatch gradient at w_k
                self.zero_grad()
                set_params(params, self.state["w"])
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

                # update the accumulator A_t
                self._update_accumulator(grad_current, grad_tilde, t)
                # record the (initial) step size
                if self.state.get("initial_step_size") is None:
                    self.state["initial_step_size"] = eta / self.state["A"]
                self.state["step_size"] = eta / self.state["A"]

                with torch.no_grad():
                    # compute x_{t+1}
                    for p_next, p_current, g_current, g_tilde, g_full in zip(
                        params,
                        params_current,
                        grad_current,
                        grad_tilde,
                        self.state["full_grad"],
                    ):
                        p_next.data = p_current.data - eta / self.state["A"] * (
                            g_current - g_tilde + g_full
                        )
                    projection_ball(params, R)

                    # record average
                    for w_record_next, w_record_curremt, x_next in zip(
                        w_record, w_record, params
                    ):
                        w_record_next.data = (
                            1 - 1 / (t + 2)
                        ) * w_record_curremt.data + (1 / (t + 2)) * x_next.data

                t += 1
                if t >= (self.state["steps_per_epoch"] - 1):
                    # update w_bar and w
                    for w_bar_next, w_bar_curremt, w in zip(
                        self.state["w_bar"], self.state["w_bar"], w_record
                    ):
                        w_bar_next.data = (1 - 1 / (k + 1)) * w_bar_curremt.data + (
                            1 / (k + 1)
                        ) * w.data
                    self.state["w"] = w_record

        # set params to w_bar for evaluation
        set_params(params, self.state["w_bar"])

        self.state["step"] += 1
        if self.state["increase_steps_per_epoch"]:
            self.state["steps_per_epoch"] = 2 ** (self.state["step"] + 2)

        return None


# helper functions
def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]


def set_params(params, params_next):
    for p, p_next in zip(params, params_next):
        p.data = p_next.data


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
