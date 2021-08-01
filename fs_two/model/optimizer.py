import copy
import random
import math
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .ranger21 import Ranger21


class PCGrad:
    def __init__(self, optimizer):
        self._optim = optimizer

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """
        clear the gradient of the parameters
        """

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """
        update the parameters with the gradient
        """

        return self._optim.step()

    def pc_backward(self, objectives):
        """
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        """

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(
            dim=0
        )
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(
            dim=0
        )
        return merged_grad

    def _set_grad(self, grads):
        """
        set the modified gradients to the network
        """

        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        pack the gradient of the parameters of the network for each objective
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx : idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        get the gradient of the parameters of the network with specific
        objective
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        """

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization """
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(
                    -x.mean(
                        dim=tuple(range(1, len(list(x.size())))), keepdim=True
                    )
                )
        else:
            if len(list(x.size())) > 1:
                x.add_(
                    -x.mean(
                        dim=tuple(range(1, len(list(x.size())))), keepdim=True
                    )
                )
    return x


class Ranger(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,  # lr
        alpha=0.5,
        k=6,
        N_sma_threshhold=5,  # Ranger options
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,  # Adam options
        # Gradient centralization on or off, applied to conv layers only or conv + fc layers
        use_gc=True,
        gc_conv_only=False,
        gc_loc=True,
    ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        if not lr > 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        if not eps > 0:
            raise ValueError(f"Invalid eps: {eps}")

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            step_counter=0,
            betas=betas,
            N_sma_threshhold=N_sma_threshhold,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        # self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}"
        )
        if self.use_gc and self.gc_conv_only == False:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_conv_only == True:
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        "Ranger optimizer does not support sparse gradients"
                    )

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if (
                    len(state) == 0
                ):  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state["slow_buffer"] = torch.empty_like(p.data)
                    state["slow_buffer"].copy_(p.data)

                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(
                        p_data_fp32
                    )

                # begin computations
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(
                        grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )

                state["step"] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state["step"] % 10)]

                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (
                        1 - beta2_t
                    )
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group["weight_decay"] != 0:
                    G_grad.add_(p_data_fp32, alpha=group["weight_decay"])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(
                        G_grad,
                        use_gc=self.use_gc,
                        gc_conv_only=self.gc_conv_only,
                    )

                p_data_fp32.add_(G_grad, alpha=-step_size * group["lr"])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state["step"] % group["k"] == 0:
                    # get access to slow param tensor
                    slow_p = state["slow_buffer"]
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = PCGrad(
            Ranger21(
                model.parameters(),
                lr=5e-4,
                betas=train_config["optimizer"]["betas"],
                eps=train_config["optimizer"]["eps"],
                weight_decay=train_config["optimizer"]["weight_decay"],
                use_adabelief=True,
                
            )
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(
            model_config["transformer"]["encoder_hidden"], -0.5
        )

    def update_lr(self):
        self._update_learning_rate()

    def pc_backward(self, losses):
        self._optimizer.pc_backward(losses)

    def real_step(self):
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.optimizer.load_state_dict(path)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.optimizer.param_groups:
            param_group["lr"] = lr
