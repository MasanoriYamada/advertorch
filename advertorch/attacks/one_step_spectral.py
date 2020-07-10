# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp
from advertorch.utils import is_float_or_torch_tensor

from .base import Attack
from .base import LabelMixin


def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs.
    it strongly assume that first axis of inputs and outputs is batch direction

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size()[1:] + inputs.size()).reshape(-1, *inputs.size())
    outputssum = torch.sum(outputs, dim=0)
    for i, out in enumerate(outputssum.view(-1)):
        col_i = autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()
    jac = jac.reshape(outputs.size()[1:] + inputs.size())
    aa = np.array(range(jac.dim()))
    cut_point = len(outputs.size())
    new_axis = np.concatenate([np.roll(aa[:cut_point], 1), aa[cut_point:]])  # batchを先頭に
    return jac.permute(tuple(new_axis))


class OSSAL2Attack(Attack, LabelMixin):
    """
    One step spectral attack (Zhao et al, 2019).
    Paper: https://arxiv.org/abs/1810.03806

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param eigen_iter: the number of iteration inthe eigen value search
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(
            self, predict, loss_fn=None, eps=2.1, eigen_iter=3,
            clip_min=0., clip_max=1., targeted=False):
        """
        Create an instance of the OSSAL2ttack.
        """
        super(OSSAL2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.predict = predict
        self.eps = eps
        self.eigen_iter = eigen_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, data, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        softmax = torch.nn.Softmax(dim=1)
        data.requires_grad = True
        out = self.predict(data)
        detached_target = out.detach()
        py = softmax(detached_target)
        log_p_y_x = logsoftmax(out)
        gy = jacobian(outputs=log_p_y_x, inputs=data, create_graph=False)
        gy_shape = gy.shape
        gy = gy.reshape(gy_shape[0], gy_shape[1], -1)
        pis, piv = self.power_iteration(py, gy, self.eigen_iter)
        eta = piv.reshape(gy_shape[0], *gy_shape[2:])
        idx = self.loss_fn(self.predict(data + eta), target) <= self.loss_fn(self.predict(data), target)
        eta[idx] = -eta[idx]
        xadv = data + self.eps * eta
        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        return xadv

    def power_iteration(self, py, gy, num_simulations: int):
        eta = torch.randn(size=(gy.shape[0], gy.shape[2])).to(gy.device)
        for _ in range(num_simulations):
            eta = torch.einsum('byi,bi,byj->byj', gy, eta, gy)
            eta = torch.einsum('by,byj->bj', py, eta)
            eta = normalize_by_pnorm(eta)
        gyeta = torch.einsum('byi,bi->by', gy, eta)
        val = torch.einsum('by,by->b', py, gyeta ** 2)
        return val, eta
