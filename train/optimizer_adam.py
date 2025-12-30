import random
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import math


def unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def norm(v, dim: int = 1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out


def Cayley_loop(X, W, tan_vec, t):  #
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))

    return Y.t()


def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    [p, n] = tan_vec.size()
    tan_vec.t_()

    # if not torch.isfinite(tan_vec).all():
    #     print("tan_vec contains NaN or Inf!")
    # print("tan_vec shape:", tan_vec.shape)
    # print("tan_vec dtype:", tan_vec.dtype)
    # print(torch.cuda.memory_summary())

    q, r = torch.linalg.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()

    return q


episilon = 1e-8


class AdamG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by Adam-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): a small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(
        self, 
        params, 
        lr, 
        momentum: int = 0, 
        dampening: int = 0,
        weight_decay: int = 0,
        nesterov: bool = False, 
        stiefel: bool = False,
        beta2: int = 0.999, 
        epsilon: float = 1e-8, 
        omega: int = 0, 
        grad_clip=None,
    ) -> None:
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            dampening=dampening,
            weight_decay=weight_decay, 
            nesterov=nesterov, 
            stiefel=stiefel,
            beta2=beta2, 
            epsilon=epsilon, 
            omega=0, 
            grad_clip=grad_clip
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']
            
            for p in group['params']:
                if p.grad is None:
                    continue
            
                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']
                lr = group["lr"]

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1,101)
                    if rand_num==1:
                        unity = qr_retraction(unity)
                        
                    g = p.grad.data.view(p.size()[0],-1)

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size=p.size()
                        param_state['m_buffer'] = torch.zeros([int(np.prod(size[1:])), size[0]])
                        param_state['v_buffer'] = torch.zeros([1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1*m  + (1.0-beta1)*g.t() # p by n
                    vnew = beta2*v  + (1.0-beta2)*(torch.norm(g)**2)
                    
                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)
                    
                    MX = torch.matmul(mnew_hat, unity)
                    XMX = torch.matmul(unity, MX)
                    XXMX = torch.matmul(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = (W_hat - W_hat.t())/vnew_hat.add(epsilon).sqrt()
                    
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)                    
                    alpha = min(t, group['lr'])
                    
                    p_new = Cayley_loop(unity.t(), W, mnew, -alpha)

                    p.data.copy_(p_new.view(p.size()))
                    mnew = torch.matmul(W, unity.t()) * vnew_hat.add(epsilon).sqrt() * (1 - beta1_power)
                    m.copy_(mnew)
                    v.copy_(vnew)

                    param_state['beta1_power']*=beta1
                    param_state['beta2_power']*=beta2
                    
                else:
                    grad = p.grad.data

                    # Adam 不支持稀疏梯度
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients")

                    state = self.state[p]

                    # ---------- state 初始化 ----------
                    if len(state) == 0:
                        state["step"] = 0.0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    t = state["step"]

                    # ---------- weight decay（L2 正则） ----------
                    if weight_decay != 0:
                        grad = grad.add(p.data, alpha=weight_decay)

                    # ---------- 一阶、二阶动量 ----------
                    # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # 需使用 exp_avg.lerp_(grad, 1 - beta1) 才能保证和官方 Adam 优化器精度一致，exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) 会导致精度不一致
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # ---------- bias correction ----------
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t

                    step_size = lr / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)

                    # ---------- 参数更新 ----------
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(epsilon)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss      