import copy

import numpy as np
import torch
import tqdm


class AdaVRAE(torch.optim.Optimizer):
    """
    Implements AdaVRAE algorithm for convex optimization.
    https://proceedings.mlr.press/v162/liu22o/liu22o.pdf
    In this implementation, we only consider constaint optimization problems
    defined within a ball centered at the origin
    Therefore, the proximal mapping becomes the projection onto the ball.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        R (float): radius of the ball for projection. If None, no projection is performed
        eta (float): eta parameter. (according to the theory, the best eta is of order D)
        train_loader (torch.utils.data.dataloader.DataLoader): Dataloader for the training data
        model_base (torch.nn.Module): Base model for computing the full gradient
        loss_function (function handle): Loss function
        replacement (bool, optional): sampling with/without replacement
        steps_per_epoch (int, optional):
            If set to None, it is n by default, where n is the number of minibatches in the training dataset
            If set to an int, it represents the customized number of steps per epoch.
    """

    def __init__(
        self,
        params,
        R,
        train_loader,
        model_base,
        loss_function,
        eta=None,
        replacement=True,
        steps_per_epoch=None,
    ):
        if not ((R is None) or (0.0 < R)):
            raise ValueError("Invalid R: {}".format(R))
        if not ((eta is None) or (0.0 < eta)):
            raise ValueError("Invalid eta: {}".format(eta))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))

        # get n
        batch_size = train_loader.batch_size
        n = len(train_loader.dataset) // float(batch_size)
        if steps_per_epoch is None:
            steps_per_epoch = n

        if not (1.0 <= steps_per_epoch):
            raise ValueError(
                "Invalid number of steps per epoch: {}. \
                perhaps the batchsize is larger than the number of training samples".format(
                    steps_per_epoch
                )
            )

        if eta is None:
            eta = 2 * R

        defaults = dict(
            R=R,
            eta=eta,
            s0=np.ceil(np.log2(np.log2(4 * n))),
            n=n,
            replacement=replacement,
            steps_per_epoch=steps_per_epoch,
        )

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["step"] = 0
        # we record 1 / gamma as th stepsize
        self.state["step_size"] = 1 / 0.01
        self.state["A"] = 5 / 4
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
            # prove convergence for u^s which will be stored in params
            self.state["u"] = copy.deepcopy(params)
            # store z_t^s as a state
            self.state["z"] = copy.deepcopy(params)
            # store g_t^s as a state
            self.state["g"] = None

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
        Update the full gradient at u
        """
        # set params to be u
        for group in self.param_groups:
            params = group["params"]
            set_params(params, self.state["u"])

        # compute the full gradient at u
        self.state["full_grad"] = self.compute_full_grad()

    def _proximal_step(self, g, step_size, R):
        """
        perform the proximal step
        """
        if step_size == np.inf:
            step_size = 1e9
        for group in self.param_groups:
            params = group["params"]
        zipped = zip(params, self.state["z"], g)
        for x_next, z_current, g in zipped:
            x_next.data = z_current.data - step_size * g
        # projection
        projection_ball(params, R)

    def compute_gamma(self, a, grad_current, grad_next):
        """
        update gamma
        """
        for group in self.param_groups:
            eta = group["eta"]
        gamma_current = 1 / self.state["step_size"]

        squared_diff = 0
        for g_current, g_next in zip(grad_current, grad_next):
            squared_diff += torch.sum(torch.mul(g_current - g_next, g_current - g_next))

        gamma_next = (
            np.sqrt((eta**2) * (gamma_current**2) + a**2 * squared_diff.item()) / eta
        )
        return gamma_next

    def compute_a(self):
        """
        compute a according to Theorem 2.1
        """
        for group in self.param_groups:
            s0 = group["s0"]
            n = group["n"]
        s = self.state["step"] + 1
        if 1 <= s <= s0:
            return (4 * n) ** (-(0.5**s))
        else:
            return (s - s0 - 1 + 3 / 2) / 3

    def step(self):
        """
        perform one inner loop of AdaVRAE
        """
        # get all the parameters
        for group in self.param_groups:
            steps_per_epoch = group["steps_per_epoch"]
            params = group["params"]
            R = group["R"]
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
                    steps_per_epoch * self.state["train_loader"].batch_size
                ),
            ),
        )
        pbar = tqdm.tqdm(loader, disable=True)
        loss_function = self.state["loss_function"]

        # update the full gradient at the very beginning
        if self.state["step"] == 0:
            self._update_memory()
            self.state["g"] = copy.deepcopy(self.state["full_grad"])
            self.state["nb_update_full_grad"] += 1

        # record the initial step size
        if self.state.get("initial_step_size") is None:
            # we record 1 / gamma instead of a / gamma
            self.state["initial_step_size"] = self.state["step_size"]

        # set xbar to be u^{s-1}
        x_bar = copy.deepcopy(self.state["u"])

        # get a
        a = self.compute_a()
        # update A
        self.state["A"] -= steps_per_epoch * (a**2)

        t = 1
        # start steps_per_epoch iterations of inner loops
        for batch in pbar:
            with torch.no_grad():
                # perform a proximal step to get x_t^s which is stored in params
                self._proximal_step(self.state["g"], a * self.state["step_size"], R)
                x_t = copy.deepcopy(params)
                # compute A_next
                A_next = self.state["A"] + a + a**2
                # update xbar
                linear_combination(
                    x_bar,
                    self.state["A"] / A_next,
                    x_bar,
                    a / A_next,
                    params,
                    (a**2) / A_next,
                    self.state["u"],
                )
                # update A
                self.state["A"] = A_next

            # update g
            if t != steps_per_epoch:
                images, labels = (
                    batch["images"].to(device=device),
                    batch["labels"].to(device=device),
                )

                self.zero_grad()
                # set params to xbar
                set_params(params, x_bar)
                # compute minibatch gradient at xbar
                _ = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    aux=self.state["aux"],
                    backwards=True,
                )
                grad_x_bar = get_grad_list(params)
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

                # compute minibatch gradient at u
                self.zero_grad()
                set_params(params, self.state["u"])
                _ = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    aux=self.state["aux"],
                    backwards=True,
                )
                grad_u = get_grad_list(params)
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

                # compute next gradient
                grad_next = [torch.zeros_like(g, requires_grad=False) for g in grad_u]
                for g_next, g_x_bar, g_u, g_full in zip(
                    grad_next, grad_x_bar, grad_u, self.state["full_grad"]
                ):
                    g_next.data = g_x_bar - g_u + g_full

            else:
                # set u to be xbar
                set_params(self.state["u"], x_bar)
                self._update_memory()
                grad_next = self.state["full_grad"]
                self.state["nb_update_full_grad"] += 1

            # update gamma
            gamma_current = 1 / self.state["step_size"]
            gamma_next = self.compute_gamma(a, self.state["g"], grad_next)

            # update z
            zipped = zip(self.state["z"], self.state["z"], x_t, grad_next)
            for z_next, z_current, xt, g_next in zipped:
                z_next.data = (
                    gamma_current / gamma_next * z_current.data
                    + (1 - gamma_current / gamma_next) * xt.data
                    - a / gamma_next * g_next.data
                )
            projection_ball(self.state["z"], R)

            # update gamma and g
            self.state["step_size"] = 1 / gamma_next
            self.state["g"] = grad_next

            t += 1
            if t > steps_per_epoch:
                # set params to u for evaluation
                set_params(params, self.state["u"])
                break

        self.state["step"] += 1
        return None


# helper functions
def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]


def set_params(params, params_next):
    for p, p_next in zip(params, params_next):
        p.data = p_next.data


def linear_combination(x, a, params_1, b, params_2, c, params_3):
    for x_next, p1, p2, p3 in zip(x, params_1, params_2, params_3):
        x_next.data = a * p1.data + b * p2.data + c * p3.data


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
