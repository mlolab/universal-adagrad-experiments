import torch
import tqdm

from universal_adagrad.base_classifier import get_classifier
from universal_adagrad.metrics import get_metric_function
from universal_adagrad.optimizer import get_optimizer


def get_model(train_loader, exp_dict, device):
    return Classifier(train_loader, exp_dict, device)


class Classifier(torch.nn.Module):
    def __init__(self, train_loader, exp_dict, device):
        super().__init__()
        self.exp_dict = exp_dict
        self.device = device

        # Load classifier and loss function
        self.model_base = get_classifier(exp_dict["model"], train_loader.dataset)
        self.loss_function = get_metric_function(self.exp_dict["loss_func"])

        # Load Optimizer
        self.to(device=self.device)
        self.opt = get_optimizer(
            opt_dict=exp_dict["opt"],
            params=self.parameters(),
            train_loader=train_loader,
            model_base=self.model_base,
            loss_function=self.loss_function,
        )

    def val_on_testset(self, dataset, metric, name):
        self.eval()

        aux = dataset.aux
        metric_function = get_metric_function(metric)
        batch_size = self.exp_dict["batch_size"]
        loader = torch.utils.data.DataLoader(
            dataset, drop_last=False, batch_size=batch_size
        )

        score_sum = 0.0
        pbar = tqdm.tqdm(loader, disable=True)
        for batch in pbar:
            images, labels = (
                batch["images"].to(device=self.device),
                batch["labels"].to(device=self.device),
            )
            score_sum += (
                metric_function(self.model_base, images, labels, aux).item()
                * images.shape[0]
            )

        score = float(score_sum / len(loader.dataset))
        return {f"{dataset.split}_{name}": score}

    def val_on_trainset(self, dataset, metric, record_grad_norm, name):
        aux = dataset.aux
        metric_function = get_metric_function(metric)
        batch_size = self.exp_dict["batch_size"]
        loader = torch.utils.data.DataLoader(
            dataset, drop_last=False, batch_size=batch_size
        )

        # compute loss
        score_sum = 0.0
        pbar = tqdm.tqdm(loader, disable=True)
        grad_store = []

        for batch in pbar:
            self.opt.zero_grad()
            images, labels = (
                batch["images"].to(device=self.device),
                batch["labels"].to(device=self.device),
            )
            score_sum += (
                metric_function(
                    self.model_base, images, labels, aux, backwards=record_grad_norm
                ).item()
                * images.shape[0]
            )

            if record_grad_norm:
                k = 0
                for p in self.model_base.parameters():
                    if len(grad_store) == 0:
                        grad_store = [
                            torch.zeros_like(p.grad, requires_grad=False)
                            for p in self.model_base.parameters()
                        ]
                    grad_store[k] += p.grad * images.shape[0]
                    k += 1

        score = float(score_sum / len(dataset))

        if record_grad_norm:
            # compute grad norm
            grad_norm2 = torch.tensor([0]).double()
            for g in grad_store:
                if g is None:
                    continue
                g /= len(dataset)
                grad_norm2 += torch.sum(torch.mul(g, g))
            grad_norm = torch.sqrt(grad_norm2)
        else:
            grad_norm = torch.tensor([-1])

        return {f"{dataset.split}_{name}": score}, {"grad_norm": grad_norm.item()}

    def get_state_dict(self):
        state_dict = {
            "model": self.model_base.state_dict(),
            "opt": self.opt.state_dict(),
        }

        return state_dict

    def set_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])
        self.opt.state["model_base"] = self.model_base

    def train_one_epoch(self):
        self.train()
        self.opt.step()

        return None
