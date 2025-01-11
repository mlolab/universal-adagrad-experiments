import copy

import numpy as np
import torch
import tqdm


class AdaVRAG(torch.optim.Optimizer):
    """
    Implements AdaVRAG algorithm for convex optimization.
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
        update_rule (int, optional):
            If set to 1, using Option I in the paper
            If set to 2, using Option II in the paper
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
        update_rule=1,
    ):
        if not ((R is None) or (0.0 < R)):
            raise ValueError("Invalid R: {}".format(R))
        if not ((eta is None) or (0.0 < eta)):
            raise ValueError("Invalid eta: {}".format(eta))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))
        if update_rule not in [1, 2]:
            raise ValueError("Invalid update rule: {}".format(update_rule))

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
            update_rule=update_rule,
        )

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["step"] = 0
        # we record 1 / gamma as th stepsize
        self.state["step_size"] = 1 / 0.01
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
            # store x_t^s as a state
            self.state["x"] = copy.deepcopy(params)
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
        zipped = zip(params, self.state["x"], g)
        for x_next, x_current, g in zipped:
            x_next.data = x_current.data - step_size * g
        # projection
        projection_ball(params, R)

    def _update_gamma(self, x_current, x_next):
        """
        update gamma
        """
        for group in self.param_groups:
            eta = group["eta"]
            update_rule = group["update_rule"]
        gamma_current = 1 / self.state["step_size"]

        squared_diff = 0.0
        for xc, xn in zip(x_current, x_next):
            squared_diff += torch.sum(torch.mul(xc.data - xn.data, xc.data - xn.data))

        if update_rule == 1:
            gamma_next = gamma_current * np.sqrt(1 + squared_diff.item() / (eta**2))
        else:
            gamma_next = gamma_current + squared_diff.item() / (eta**2)

        self.state["step_size"] = 1 / gamma_next

    def compute_a_and_q(self):
        """
        compute a and q according to Theorem 2.2
        """
        for group in self.param_groups:
            s0 = group["s0"]
            n = group["n"]
        s = self.state["step"] + 1
        if 1 <= s <= s0:
            a = 1 - (4 * n) ** (-(0.5**s))
            q = 1 / ((1 - a) * a)
        else:
            c = (3 + np.sqrt(33)) / 4
            a = c / (s - s0 + 2 * c)
            q = 8 * (2 - a) * a / (3 * (1 - a))
        return a, q

    def step(self):
        """
        perform one inner loop of AdaVRAG
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

        # record the initial step size
        if self.state.get("initial_step_size") is None:
            # we record 1 / gamma
            self.state["initial_step_size"] = self.state["step_size"]

        # compute a and q
        a, q = self.compute_a_and_q()

        # initialize xbar_average as zero
        xbar_average = copy.deepcopy(params)
        for p in xbar_average:
            p.data = torch.zeros_like(p.data, requires_grad=False)
        # initialize xbar
        xbar = copy.deepcopy(xbar_average)
        with torch.no_grad():
            # update xbar
            linear_combination(xbar, a, self.state["x"], 1 - a, self.state["u"])

        # update full gradient at u
        self._update_memory()
        self.state["nb_update_full_grad"] += 1

        t = 1
        # start steps_per_epoch iterations of inner loops of AdaVRAG
        for batch in pbar:
            images, labels = (
                batch["images"].to(device=device),
                batch["labels"].to(device=device),
            )

            # compute minibatch gradient at xbar
            self.zero_grad()
            set_params(params, xbar)
            _ = loss_function(
                self.state["model_base"],
                images,
                labels,
                aux=self.state["aux"],
                backwards=True,
            )
            grad_xbar = get_grad_list(params)
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

            # compute x_next and store it in params
            grad_vr = [torch.zeros_like(g, requires_grad=False) for g in grad_u]
            for g_vr, g_xbar, g_u, g_full in zip(
                grad_vr, grad_xbar, grad_u, self.state["full_grad"]
            ):
                g_vr.data = g_xbar - g_u + g_full

            self._proximal_step(grad_vr, 1 / q * self.state["step_size"], R)

            # update gamma
            self._update_gamma(self.state["x"], params)

            # update x
            set_params(self.state["x"], params)

            with torch.no_grad():
                # update xbar
                linear_combination(xbar, a, self.state["x"], 1 - a, self.state["u"])
                # update xbar_average
                linear_combination(xbar_average, 1 / t, xbar, 1 - 1 / t, xbar_average)

            t += 1
            if t > steps_per_epoch:
                # set u to be xbar_average
                set_params(self.state["u"], xbar_average)
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
