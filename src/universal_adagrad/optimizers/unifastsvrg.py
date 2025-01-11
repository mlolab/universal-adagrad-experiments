import copy

import numpy as np
import torch
import tqdm


class UniFastSVRG(torch.optim.Optimizer):
    """
    Implements UniFastSvrg (Alg. 4) for convex optimization.
    In this implementation, we only consider constaint optimization problems
    defined within a ball centered at the origin
    Therefore, the proximal mapping becomes the projection onto the ball.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        R (float): radius of the ball for projection. If None, no projection is performed
        A0 (float): coefficient A_0 for the standard AccSVRG (A0 ~ 1 / steps_per_epoch)
        train_loader (torch.utils.data.dataloader.DataLoader): Dataloader for the training data
        model_base (torch.nn.Module): Base model for computing the full gradient
        loss_function (function handle): Loss function
        replacement (bool, optional): sampling with/without replacement
        steps_per_epoch (int, optional):
            If set to None, it is m by default, where m is the number of minibatches in the training dataset
            If set to an int, it represents the customized number of steps per epoch.
        stepsize_rule (string optinal): 'new', 'old' or 'AdaGrad'
        a_rule (int optional): [1,2,3]
        initial_proximal_step (bool optional): whether or not do the initial proximal step for xtilde
        count_cheat (bool, optional): This option is for debugging purpose.
            if set to True, we dont count the extra gradient evaluations when updating the stepsizes,
            By default, it is set to False.
    """

    def __init__(
        self,
        params,
        R,
        train_loader,
        model_base,
        loss_function,
        A0=None,
        replacement=True,
        steps_per_epoch=None,
        stepsize_rule="new",
        a_rule=3,
        initial_proximal_step=False,
        count_cheat=False,
    ):
        if not ((A0 is None) or (0.0 < A0)):
            raise ValueError("Invalid A0: {}".format(A0))
        if not ((R is None) or (0.0 < R)):
            raise ValueError("Invalid R: {}".format(R))
        if not isinstance(replacement, bool):
            raise ValueError("Invalid replacement: {}".format(replacement))
        if not isinstance(count_cheat, bool):
            raise ValueError("Invalid count_cheat: {}".format(count_cheat))
        if not isinstance(initial_proximal_step, bool):
            raise ValueError(
                "Invalid initial proximal step: {}".format(initial_proximal_step)
            )
        if stepsize_rule not in ["new", "old", "AdaGrad"]:
            raise ValueError("Invalid stepsize_rule: {}".format(stepsize_rule))
        if a_rule not in [1, 2, 3]:
            raise ValueError("Invalid stepsize_rule: {}".format(a_rule))

        # get m
        batch_size = train_loader.batch_size
        m = len(train_loader.dataset) // float(batch_size)
        if steps_per_epoch is None:
            steps_per_epoch = m

        if not (1.0 <= steps_per_epoch):
            raise ValueError(
                "Invalid number of steps per epoch: {}. \
                perhaps the batchsize is larger than the number of training samples".format(
                    steps_per_epoch
                )
            )

        if A0 is None:
            A0 = 1 / m

        defaults = dict(
            R=R,
            replacement=replacement,
            steps_per_epoch=steps_per_epoch,
            stepsize_rule=stepsize_rule,
            a_rule=a_rule,
        )

        super().__init__(params, defaults)

        self.state["train_loader"] = train_loader
        self.state["model_base"] = model_base
        self.state["loss_function"] = loss_function
        self.state["aux"] = train_loader.dataset.aux
        self.state["count_cheat"] = count_cheat
        self.state["step"] = 0
        self.state["step_size"] = np.inf
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
            if initial_proximal_step:
                # initialize x_tilde to be prox(x0,g0,0)
                # compute full gradient at x0
                self._update_memory()
                for x_tilde_next, x_current, g_full in zip(
                    self.state["x_tilde"], params, self.state["full_grad"]
                ):
                    x_tilde_next.data = x_current.data - 1e8 * g_full
                # projection
                projection_ball(self.state["x_tilde"], R)
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

    def update_stepsize(
        self, a, y_current, v_current, grad_y, grad_tilde, stepsize_rule
    ):
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
                # compute stochastic loss value and minibatch gradient at y_current
                self.zero_grad()
                set_params(params, y_current)
                loss_y_prime = loss_function(
                    self.state["model_base"],
                    images,
                    labels,
                    self.state["aux"],
                    backwards=True,
                )
                grad_y_prime = get_grad_list(params)
                if not self.state["count_cheat"]:
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
            if not self.state["count_cheat"]:
                self.state["nb_forwards"] += 1
                self.state["nb_backwards"] += 1

            break

        if stepsize_rule in ["old"]:
            # compute brgeman distance between xplus and y
            breg_dist_y_xplus_prime = compute_bregman_distance(
                loss_xplus_prime, loss_y_prime, grad_y_prime, y_current, xplus
            )

            breg_dist_sum = 0.0

            # compute <G_1 - G, vplus - v>
            diff_inner_prod = 0.0
            for g_y, g_tilde, g_y_prime, g_tilde_prime, pplus, p in zip(
                grad_y,
                grad_tilde,
                grad_y_prime,
                grad_tilde_prime,
                self.state["v"],
                v_current,
            ):
                diff_inner_prod += torch.sum(
                    torch.mul(
                        pplus.data - p.data, g_y_prime - g_tilde_prime - g_y + g_tilde
                    )
                )

            # compute squared distance between vplus and v_current
            squared_dist = 0.0
            for pplus, p in zip(self.state["v"], v_current):
                squared_dist += torch.sum(
                    torch.mul(pplus.data - p.data, pplus.data - p.data)
                )

        elif stepsize_rule in ["new"]:
            # compute <G+ - G, xplus - y>
            # note that xplus = yplus
            diff_inner_prod = 0.0
            for g_y, g_tilde, g_xplus_prime, g_tilde_prime, pplus, p in zip(
                grad_y, grad_tilde, grad_xplus_prime, grad_tilde_prime, xplus, y_current
            ):
                diff_inner_prod += torch.sum(
                    torch.mul(
                        pplus.data - p.data,
                        g_xplus_prime - g_tilde_prime - g_y + g_tilde,
                    )
                )

            # compute squared distance between xplus and y_current
            squared_dist = 0.0
            for pplus, p in zip(xplus, y_current):
                squared_dist += torch.sum(
                    torch.mul(pplus.data - p.data, pplus.data - p.data)
                )

        if stepsize_rule in ["AdaGrad"]:
            # compute squared distance between G+ and G
            squared_dist = 0.0
            for g_y, g_tilde, g_xplus_prime, g_tilde_prime in zip(
                grad_y, grad_tilde, grad_xplus_prime, grad_tilde_prime
            ):
                squared_dist += torch.sum(
                    torch.mul(
                        g_xplus_prime - g_tilde_prime - g_y + g_tilde,
                        g_xplus_prime - g_tilde_prime - g_y + g_tilde,
                    )
                )

        # compute A+
        Aplus = self.state["A"] + a

        # compute H+
        # tentative stepsize
        if stepsize_rule in ["old"]:
            Hplus = (
                (
                    H * (D**2)
                    + Aplus * breg_dist_y_xplus_prime
                    + a * diff_inner_prod
                    - self.state["A"] / 2 * breg_dist_sum
                )
                / ((D**2) + squared_dist / 2)
            ).item()
            # check
            if (
                Aplus * breg_dist_y_xplus_prime
                + a * diff_inner_prod
                - Hplus / 2 * squared_dist
                - self.state["A"] / 2 * breg_dist_sum
            ) < 0:
                Hplus = H

        elif stepsize_rule in ["new"]:
            Hplus = (
                (H * (D**2) / Aplus + diff_inner_prod)
                / ((a**2) * (D**2) / (Aplus**2) + squared_dist / 2)
            ).item()

            # check
            if diff_inner_prod - Hplus / 2 * squared_dist < 0:
                Hplus = H
            else:
                Hplus = (a**2) * Hplus / Aplus

        elif stepsize_rule in ["AdaGrad"]:
            Hplus = np.sqrt(H**2 + (a**2) / (D**2) * squared_dist.item())

        if Hplus == 0.0:
            self.state["step_size"] = np.inf
        else:
            self.state["step_size"] = 1 / Hplus

        # set params back to xplus
        set_params(params, xplus)

        if stepsize_rule in ["new", "AdaGrad"]:
            return grad_xplus_prime, grad_tilde_prime

    def _proximal_step(self, grad_y, grad_tilde, step_size, R):
        """
        perform the proximal step
        """
        if step_size == np.inf:
            step_size = 1e9
        zipped = zip(
            self.state["v"],
            self.state["v"],
            grad_y,
            grad_tilde,
            self.state["full_grad"],
        )
        for v_next, v_current, g_y, g_tilde, g_full in zipped:
            v_next.data = v_current.data - step_size * (g_y - g_tilde + g_full)
        # projection
        projection_ball(self.state["v"], R)

    def compute_a(self, a_rule):
        """
        compute a_{t+1} from the equation:
        rule1: a_{t+1}^2 / (A_t + a_{t+1}) + 4 * a_{t+1}^2 / A_t = 1
        rule2: a_{t+1} = sqrt{A_t}
        rule3: a_{t+1}^2 / (A_t + a_{t+1}) + a_{t+1}^2 / A_t = 1

        """
        if a_rule == 2:
            return np.sqrt(self.state["A"])
        else:
            A = self.state["A"]
            if a_rule == 1:
                p = np.poly1d([4, 5 * A, -1 * A, -A * A])
            else:
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
            R = group["R"]
            replacement = group["replacement"]
            stepsize_rule = group["stepsize_rule"]
            a_rule = group["a_rule"]

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
        a = self.compute_a(a_rule)

        # update the full gradient at x_tilde
        self._update_memory()
        self.state["nb_update_full_grad"] += 1

        # set xbar to be zeros
        xbar = copy.deepcopy(params)
        for p in xbar:
            p.data = torch.zeros_like(p.data, requires_grad=False)

        k = 0

        # start steps_per_epoch iterations of Universal Triangle SVRG Steps
        for batch in pbar:
            # set params to be y
            linear_combination(
                params, A / (A + a), self.state["x_tilde"], a / (A + a), self.state["v"]
            )
            y_current = copy.deepcopy(params)

            if (k == 0 and stepsize_rule in ["new", "AdaGrad"]) or (
                stepsize_rule in ["old"]
            ):
                images, labels = (
                    batch["images"].to(device=device),
                    batch["labels"].to(device=device),
                )

                # compute minibatch gradient at y
                self.zero_grad()
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
            v_current = copy.deepcopy(self.state["v"])
            self._proximal_step(grad_y, grad_tilde, a * self.state["step_size"], R)

            # set params to be x+
            linear_combination(
                params, A / (A + a), self.state["x_tilde"], a / (A + a), self.state["v"]
            )

            # update stepsize
            if stepsize_rule in ["new", "AdaGrad"]:
                # update the next grad_y and grad_tilde
                grad_y, grad_tilde = self.update_stepsize(
                    a, y_current, v_current, grad_y, grad_tilde, stepsize_rule
                )
            else:
                self.update_stepsize(
                    a, y_current, v_current, grad_y, grad_tilde, stepsize_rule
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
                set_params(self.state["x_tilde"], xbar)

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


def compute_bregman_distance(loss_xplus, loss_x, grad_x, x, xplus):
    # return loss_xplus - loss_x - <grad_x, xplus - x>
    with torch.no_grad():
        innerprod = 0.0
        for p, pplus, g in zip(x, xplus, grad_x):
            innerprod += torch.sum(torch.mul(pplus.data - p.data, g))
    return loss_xplus - loss_x - innerprod


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
