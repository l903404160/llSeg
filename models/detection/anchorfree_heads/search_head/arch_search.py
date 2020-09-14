import torch
import numpy as np
import torch.nn as nn
from models.detection.anchorfree_heads.search_head.model_search import SearchHead
"""
    Search Controler
"""


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class SearchArch(object):
    def __init__(self, model):
        self._net_momentum = 0
        self._net_weight_decay = 1e-3
        self._model = model
        self._lr = 6e-4
        self._optimizer = torch.optim.Adam(self._model.arch_parameters(), lr=self._lr,
                                          betas=(0.5, 0.999), weight_decay=self._net_weight_decay)

    def search_step(self, data, val_data, eta, network_optimizer, unrolled):

        self._optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(data, val_data, eta, network_optimizer)
        else:
            self._backward_step(data)

        self._optimizer.step()

    def _backward_step(self, data):
        """
        Args:
            input_valid:
            target_valid:
        Returns:
        """
        loss = self._model.model_forward(data)
        loss.backward()

    # update model parameters by second order gradient.
    def _backward_step_unrolled(self, data, val_data, eta, net_optimizer):

        unrolled_model = self._compute_unrolled_model(data, eta, net_optimizer)
        unrolled_loss = unrolled_model.model_forward(val_data)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, data)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self._model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = torch.autograd.Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _compute_unrolled_model(self, data, eta, net_optimizer):
        loss = self._model.model_forward(data)

        theta = _concat(self._model.parameters()).data
        try:
            moment = _concat(net_optimizer.state[v]['momentum_buffer'] for v in self._model.parameters()).mul_(self._net_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self._model.parameters())).data + self._net_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _construct_model_from_theta(self, theta):
        # TODO new
        model_new = self._model.new()
        model_dict = self._model.state_dict()

        params, offset = {}, 0
        for k,v in self._model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r / _concat(vector).norm()
        for p,v in zip(self._model.parameters(), vector):
            p.data.add_(R, v)
        loss = self._model.model_forward(data)

        grads_p = torch.autograd.grad(loss, self._model.arch_parameters())

        for p,v in zip(self._model.parameters(), vector):
            p.data.sub_(2*R, v)

        loss = self._model.model_forward(data)

        grads_n = torch.autograd.grad(loss, self._model.arch_parameters())

        for p,v in zip(self._model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    data_path = '/home/haida_sunxin/lqx/data/search/000000190236.pth'
    data = torch.load(data_path)

    m = SearchHead(C_in=256, C_mid=256, C_out=256, num_classes=80, layers=1, multiplier=2).cuda()
    optimizer = torch.optim.SGD(m.parameters(), lr=0.01, weight_decay=3e-4, momentum=0.9)
    print(m)

    a = SearchArch(m)

    features = {
        'p3': data['p3'].unsqueeze(0).cuda(),
        'p4': data['p4'].unsqueeze(0).cuda(),
        'p5': data['p5'].unsqueeze(0).cuda(),
        'p6': data['p6'].unsqueeze(0).cuda(),
        'p7': data['p7'].unsqueeze(0).cuda(),
    }

    targets = {
        'labels':data['labels'].unsqueeze(0).cuda(),
        'reg_targets':data['reg_targets'].unsqueeze(0).cuda(),
        'ctr_targets':data['ctr_targets'].unsqueeze(0).cuda()
    }

    data = {
        'features': features,
        'targets': targets
    }

    import time

    st = time.time()
    for i in range(50):
        if i > 20:
            a.search_step(data, data, eta=0.01, network_optimizer=optimizer, unrolled=True)
            if (i+1) % 10 == 0:
                print(torch.nn.functional.softmax(m.arch_parameters()[0], dim=-1))
                print(torch.nn.functional.softmax(m.arch_parameters()[1], dim=-1))


        loss = m.model_forward(data, flag=True)
        print(loss)
        loss = sum(loss.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    name = '/home/haida_sunxin/lqx/data/search/000000190236.pth'
    out_dir = '/home/haida_sunxin/lqx'
    m.visualization(name, out_dir)

    print(m.arch_parameters())
    print(m.genotype())
    print('using %f s' % (time.time() - st))
    print('done')