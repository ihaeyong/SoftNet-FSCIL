import torch
import json

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGDHessian(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t-1} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, freq_path=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # freq path
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, sample_per_class=freq)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGDHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def __setstate__(self, state):
        super(SGDHessian, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    def init(self):
        self.generator = torch.Generator().manual_seed(2147483647)

        for group in self.param_groups:
            self.n_samples = 1
            self.update_each = 1

        for p in self.get_params():
            p.hess = torch.zeros_like(p)
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """
        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self, p, g):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        if False:
            for p in filter(lambda p: p.grad is not None, self.get_params()):
                if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                    params.append(p)
                self.state[p]["hessian step"] += 1

        else:
            params.append(p)

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        if False:
            grads = [p.grad for p in params]
        else:
            grads = [g]

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i <= self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)

        return p.hess


    def sgd_step(self, params, d_p_list, momentum_buffer_list, *,
                 weight_decay: float,
                 momentum: float,
                 lr: float,
                 dampening: float,
                 nesterov: bool,
                 maximize: bool,
                 eigen_val: float,
                 eigen_vec: float):

        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        for i, param in enumerate(params):

            d_p = d_p_list[i]
            d_h = self.set_hessian(param,d_p)
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    buf = d_p
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr

            if False:
                '''
                sgd
                '''
                param.add_(d_p, alpha=alpha)
            elif False:
                '''
                weighted sgd
                '''
                spc = self.sample_per_class.type_as(params[0])
                spc = spc.unsqueeze(0).expand(params[0].shape[1], -1).t()

                w_list = 1.0 / torch.pow(spc, 1/4)
                max_w = -alpha
                w_list = w_list * (max_w / torch.max(w_list))

                param.add_(d_p * w_list, alpha=1.0)
            elif True:
                '''
                pruned weighted sgd
                '''
                spc = self.sample_per_class.type_as(params[0])
                spc = spc.unsqueeze(0).expand(params[0].shape[1], -1).t()

                w_list = 1.0 / torch.pow(spc, 1/4)
                max_w = -lr
                w_list = w_list * (max_w / torch.max(w_list))

                if False:
                    mean_dp = d_p.mean(1)[:,None].repeat(1,d_p.size(1))
                    var_dp = d_p.var(1)[:,None].repeat(1,d_p.size(1))

                    mask_pos = (d_p > (mean_dp)).float()
                    mask_neg = (d_p <= (mean_dp)).float()
                elif True:
                    mean_dp = d_p.abs().mean(1)[:,None].repeat(1,d_p.size(1))
                    var_dp = d_p.abs().var(1)[:,None].repeat(1,d_p.size(1))

                    th = mean_dp - 0.01
                    mask_pos = (d_p.abs() > th).float()
                    mask_neg = (d_p.abs() <= th).float()

                    p_th = param.abs().mean()
                    mask_p_pos = (param.abs() > p_th).float()
                    mask_p_neg = (param.abs() <= p_th).float()
                else:
                    th = 1e-3
                    mask_pos = (d_p > th).float()
                    mask_neg = (d_p <= th).float()

                if eigen_vec and False:
                    hlr1 = lr * 1e-5
                    hlr2 = lr * 1e-1
                    if False:
                        d_p_update += hlr1 * eigen_vec[0][0] + hlr2 * eigen_vec[1][0]
                    elif False:
                        d_p_update = d_p + hlr2 * eigen_vec[1][0].data
                    elif True:
                        eigen1 = hlr1 * eigen_vec[0][0].data
                        eigen2 = hlr2 * eigen_vec[1][0].data

                        d_p_pos = d_p * mask_pos * 0.9
                        if True:
                            d_p_neg = d_p * mask_neg - eigen2 * w_list
                        else:
                            d_p_neg = d_p * mask_neg + eigen2 * d_h
                        d_p_update = d_p_pos + d_p_neg
                    elif False:
                        d_p_update = d_p + hlr2 * eigen_vec[1][0].data

                else:
                    d_p_pos = d_p * mask_pos * 0.9
                    d_p_neg = d_p * mask_neg
                    d_p_update = d_p_pos + d_p_neg


                if True:
                    param.add_(d_p_update * w_list, alpha=1.0)
                elif False:
                    param.add_(d_p_update, alpha=1.0)
                else:
                    param.add_(d_p_update, alpha=alpha)

    @torch.no_grad()
    def step(self, closure=None, eigen_val=None, eigen_vec=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.init()
        self.zero_hessian()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.sgd_step(params_with_grad,
                          d_p_list,
                          momentum_buffer_list,
                          weight_decay=weight_decay,
                          momentum=momentum,
                          lr=lr,
                          dampening=dampening,
                          nesterov=nesterov,
                          maximize=maximize,
                          eigen_val=eigen_val,
                          eigen_vec=eigen_vec)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
