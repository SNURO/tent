from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import ipdb


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, div_reg=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.div_reg = div_reg

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, div_reg=self.div_reg)

        return outputs
    
    def forward_only(self, x):
        if self.episodic:
            self.reset()

        return self.model(x)

    def forward_gt(self, x, y):
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            outputs = forward_and_adapt_gt(x, y, self.model, self.optimizer, self.loss_fn)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    def set_test(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def set_div_reg(self, bool: bool):
        self.div_reg = bool

class PL(Tent):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__(model, optimizer, steps, episodic)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt_pl(x, self.model, self.optimizer, self.loss_fn)

        return outputs
    

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, div_reg=False):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    if div_reg:
        msoftmax = outputs.softmax(1).mean(dim=0)
        # below is not exactly entropy, no minus sign
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-16))
        loss = loss + gentropy_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

#TODO0803: 만약 div_reg를 gt나 pl에도 적용하고 싶으면 위의 div_reg code 비슷하게 넣어야 함

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_gt(x, y, model, optimizer, loss_fn):
    """Forward and adapt model on batch of data. by gt labels

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    y = y.to(torch.int64)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_pl(x, model, optimizer, loss_fn):
    """Forward and adapt model on batch of data. instead of PL not ENTROPY

    Make pseudo labels, CE loss, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    pseudo_label = outputs.argmax(1).detach()
    loss = loss_fn(outputs, pseudo_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model

def linear_retrain(model, no_bias=False, freeze_bias=False):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    #ipdb.set_trace()
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.Linear):
            for np, p in m.named_parameters():
                if np in ['weight']:
                    p.requires_grad_(True)
                elif np in ['bias'] and freeze_bias==False:
                    p.requires_grad_(True)
                elif np in ['bias'] and no_bias==True:
                    p.data = torch.zeros_like(p.data)


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
