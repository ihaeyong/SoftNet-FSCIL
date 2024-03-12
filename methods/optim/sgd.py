import torch
import json

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGD(torch.optim.Optimizer):
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
                 weight_decay=0, nesterov=False, *, maximize=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))


        self.sample_per_class = None

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    def sgd_v_step(self, params, d_p_list, momentum_buffer_list, *,
                   weight_decay: float,
                   momentum: float,
                   lr: float,
                   dampening: float,
                   nesterov: bool,
                   maximize: bool,
                   mask: float,
                   mask_v: float):

        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        assert len(params) == len(mask)

        for i, (param, m_, m_v) in enumerate(zip(params, mask, mask_v)):

            d_p = d_p_list[i]
            if weight_decay != 0 and d_p is not None:
                d_p = d_p.add(param, alpha=weight_decay)
            if momentum != 0:
                buf = momentum_buffer_list[i]
                if buf is None :
                    if d_p is not None:
                        buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    if d_p is not None:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov :
                    if d_p is not None:
                        d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr

            # sgd
            if False:
                param.add_(d_p, alpha=alpha)
            else:
                if d_p is not None:
                    param[m_==1] += d_p[m_==1] * alpha
                    param[m_v==1] += d_p[m_v==1] * alpha

    def sgd_step(self, params, d_p_list, momentum_buffer_list, *,
                 weight_decay: float,
                 momentum: float,
                 lr: float,
                 dampening: float,
                 nesterov: bool,
                 maximize: bool,mask: float):

        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        assert len(params) == len(mask)

        for i, (param, m_) in enumerate(zip(params,mask)):

            d_p = d_p_list[i]
            if weight_decay != 0 and d_p is not None:
                d_p = d_p.add(param, alpha=weight_decay)
            if momentum != 0:
                buf = momentum_buffer_list[i]
                if buf is None :
                    if d_p is not None:
                        buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    if d_p is not None:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov :
                    if d_p is not None:
                        d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr

            # sgd
            if False:
                param.add_(d_p, alpha=alpha)
            else:
                if d_p is not None:
                    param[m_==1].add_(d_p[m_==1], alpha=alpha)

    def sgd_v_step(self, params, d_p_list, momentum_buffer_list, *,
                   weight_decay: float,
                   momentum: float,
                   lr: float,
                   dampening: float,
                   nesterov: bool,
                   maximize: bool,
                   mask: float,
                   mask_v: float):

        r"""Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """

        assert len(params) == len(mask)

        for i, (param, m_, m_v) in enumerate(zip(params, mask, mask_v)):

            d_p = d_p_list[i]
            if weight_decay != 0 and d_p is not None:
                d_p = d_p.add(param, alpha=weight_decay)
            if momentum != 0:
                buf = momentum_buffer_list[i]
                if buf is None :
                    if d_p is not None:
                        buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    if d_p is not None:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov :
                    if d_p is not None:
                        d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            alpha = lr if maximize else -lr

            # sgd
            if False:
                param.add_(d_p, alpha=alpha)
            else:
                if d_p is not None:
                    param[m_==1].add_(d_p[m_==1], alpha=alpha)
                    param[m_v==1].add_(d_p[m_v==1], alpha=alpha * 1e-2)

    @torch.no_grad()
    def step(self, closure=None, mask=None, mask_v=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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
            for i in range(len(group['params'])):
                p = group['params'][i]
                #if p.grad is not None:
                if p.requires_grad :
                    #print("grad_idx:{},{}".format(i, p.shape))
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                else:
                    None
                    #print("non_grad_idx:{},{}".format(i, p.shape))

            if mask_v is not None:
                self.sgd_v_step(params_with_grad,
                                d_p_list,
                                momentum_buffer_list,
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr,
                                dampening=dampening,
                                nesterov=nesterov,
                                maximize=maximize,
                                mask=mask,
                                mask_v=mask_v)
            else:
                self.sgd_step(params_with_grad,
                              d_p_list,
                              momentum_buffer_list,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              lr=lr,
                              dampening=dampening,
                              nesterov=nesterov,
                              maximize=maximize,
                              mask=mask)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
