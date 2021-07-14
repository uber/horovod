from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size,  rank, local_rank

import mxnet as mx
from collections import OrderedDict, defaultdict
import types
import time
import warnings
from mxnet.gluon.parameter import Parameter


class POS_Trainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None,
                 gradient_predivide_factor=1.0, prefix=None):

        self._world_size = size()
        self._world_rank = rank()

        self._all_params = []
        self._all_param2idx = {}
        param_list = []
        if isinstance(params, (dict, OrderedDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s." % (type(params)))
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s." % (type(param)))
            if param._uuid in self._all_param2idx:
                # Shared parameters have same uuid; only need to store one of the shared versions
                continue
            self._all_param2idx[param._uuid] = i
            self._all_params.append(param)
        self._partition_params, self._param2rank = self._partition_parameters(self._all_params)
        self._own_part = self._partition_params[self._world_rank]
        super(POS_Trainer, self).__init__(
            self._own_part, optimizer, optimizer_params=optimizer_params, kvstore=None)
        self._prefix = prefix if prefix else ""
        self._scale = gradient_predivide_factor / size()
        self._gradient_predivide_factor = gradient_predivide_factor


    def _partition_parameters(self, params):
        """
        partition all the parameters by their size and try to average them.
        """
        world_size = self._world_size
        ## list for rank each would be
        partition_params = [[] for _ in range(world_size)]
        param2rank = {}
        sizes = [0 for _ in range(world_size)]
        for param in params:
            if param.grad_req != 'null':
                current_rank = sizes.index(min(sizes))
                partition_params[current_rank].append(param)
                num = 1
                param2rank[param._uuid] = current_rank
                for p in param.shape:
                    num *= p
                sizes[current_rank] += num
        return partition_params, param2rank

    def _allreduce_grads(self):
        """
        rewrite allreduce here because we need to communicate using horovod.
        Actually we should use scatter here, but since it is not available yet,
        I use allreduce instead.
        """
        for i, param in enumerate(self._all_params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=self._prefix + str(i), priority=-i,
                           prescale_factor=1.0 / self._gradient_predivide_factor)

    def step(self, batch_size, ignore_stale_grad=False):
        """
        inherit from trainer, only call boardcast to make sure all parameter are consistent
        Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        Since each process main their own part, we need to brodcast after calculation
        """
        super(POS_Trainer, self).step(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()

    def update(self, batch_size, ignore_stale_grad=False):
        '''
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        assert not (self._kvstore and self._update_on_kvstore), \
            'update() when parameters are updated on kvstore ' \
            'is not supported. Try setting `update_on_kvstore` ' \
            'to False when creating trainer.'
        Since each process main their own part, we need to brodcast after calculation
        '''
        super(POS_Trainer, self).update(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()



    def _broadcast_partition_params(self):
        """
        This function is to broadcast parameter since each process will maintain their own part
        """
        for param in self._all_params:
            broadcast_(param.data(), self._param2rank[param._uuid], name=str(self._all_param2idx[param._uuid]))
