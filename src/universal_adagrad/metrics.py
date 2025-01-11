import torch


def get_metric_function(metric_name):
    if metric_name == "polyhedron_loss":
        return polyhedron_loss

    if metric_name == "logistic_loss":
        return logistic_loss

    elif metric_name == "logistic_func_gap":
        return logistic_func_gap

    elif metric_name == "hinge_loss":
        return hinge_loss

    elif metric_name == "hinge_func_gap":
        return hinge_func_gap

    elif metric_name == "huber_loss":
        return huber_loss

    elif metric_name == "huber_func_gap":
        return huber_func_gap

    elif metric_name == "L2_loss":
        return L2_loss

    elif metric_name == "L2_func_gap":
        return L2_func_gap

    elif metric_name == "class2_acc":
        return class2_accuracy

    elif metric_name == "softmax_accuracy":
        return softmax_accuracy

    elif metric_name == "softmax_loss":
        return softmax_loss


def polyhedron_loss(model, images, labels, aux=None, backwards=False):
    # return Avg((<a_i, x> - b_i)_+^q)
    # each row of A is a_i
    A = model(images)
    q = aux.get("q")
    for p in model.parameters():
        loss = (torch.nn.functional.relu(A @ p.view(-1) - labels) ** q).mean()
    if backwards and loss.requires_grad:
        loss.backward()
    return loss


def L2_loss(model, images, labels, aux=None, backwards=False):
    # return Avg(||A_i @ x - labels_i||_2) where A_i is a diagonal matrix store in a row
    # each row of A is A_i
    A = model(images)
    for p in model.parameters():
        loss = torch.sqrt(torch.mul(A * p - labels, A * p - labels).sum(axis=1)).mean()

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def L2_func_gap(model, images, labels, aux, backwards=False):
    fstar = aux["fstar"]
    loss = L2_loss(model, images, labels, aux=aux, backwards=backwards)
    return loss - fstar


def logistic_loss(model, images, labels, aux=None, backwards=False):
    logits = model(images).view(-1)
    # logits has shape (nb_samples, 1)
    loss = torch.log(1 + torch.exp(-1 * labels * logits)).mean()

    if aux["sigma2"] is not None:
        w = 0.0
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += aux["sigma2"] / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def logistic_func_gap(model, images, labels, aux, backwards=False):
    fstar = aux["fstar"]
    loss = logistic_loss(model, images, labels, aux=aux, backwards=backwards)
    return loss - fstar


def hinge_loss(model, images, labels, aux=None, backwards=False):
    logits = model(images).view(-1)
    # logits has shape (nb_samples, 1)
    loss = torch.nn.functional.relu(1 - labels * logits).mean()

    if aux["sigma2"] is not None:
        w = 0.0
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += aux["sigma2"] / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def hinge_func_gap(model, images, labels, aux, backwards=False):
    fstar = aux["fstar"]
    loss = hinge_loss(model, images, labels, aux=aux, backwards=backwards)
    return loss - fstar


def huber_loss(model, images, labels, aux, backwards=False):
    logits = model(images).view(-1)
    # logits has shape (nb_samples, 1)
    mu = aux["mu_huber"]
    loss = torch.where(
        torch.abs(1 - labels * logits) < mu,
        (1 - labels * logits) ** 2 / (2 * mu),
        torch.abs(1 - labels * logits) - mu / 2,
    ).mean()

    if aux["sigma2"] is not None:
        w = 0.0
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += aux["sigma2"] / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def huber_func_gap(model, images, labels, aux, backwards=False):
    fstar = aux["fstar"]
    loss = huber_loss(model, images, labels, aux=aux, backwards=backwards)
    return loss - fstar


def class2_accuracy(model, images, labels, aux=None, backwards=False):
    pred_labels = torch.sign(model(images)).view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc


def softmax_loss(model, images, labels, aux=None, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if aux["sigma2"] is not None:
        w = 0.0
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += aux["sigma2"] / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def softmax_accuracy(model, images, labels, aux=None, backwards=False):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
