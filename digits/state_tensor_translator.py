import copy
from collections import OrderedDict


class StateTensorTranslator:

    def __init__(self, netF_state, netB_state, netC_state):
        netF_state, netB_state, netC_state = copy.deepcopy(netF_state), \
                                             copy.deepcopy(netB_state), copy.deepcopy(netC_state)

        self.netF_keys = list(netF_state.keys())
        self.netB_keys = list(netB_state.keys())
        self.netC_keys = list(netC_state.keys())

        self.netF_shapes = [t.shape for t in netF_state.values()]
        self.netB_shapes = [t.shape for t in netB_state.values()]
        self.netC_shapes = [t.shape for t in netC_state.values()]

    def tensor_shapes(self):
        return self.netF_shapes + self.netB_shapes + self.netC_shapes

    def states_to_tensor(self, netF_state, netB_state, netC_state):
        F_values = [netF_state[k] for k in self.netF_keys]
        B_values = [netB_state[k] for k in self.netB_keys]
        C_values = [netC_state[k] for k in self.netC_keys]
        return F_values + B_values + C_values

    def tensor_to_state(self, tensors):
        tensor_index = 0

        F_state = OrderedDict()
        for i, k in enumerate(self.netF_keys):
            F_state[k] = tensors[tensor_index]
            assert self.netF_shapes[i] == tensors[tensor_index].shape
            tensor_index += 1

        B_state = OrderedDict()
        for i, k in enumerate(self.netB_keys):
            B_state[k] = tensors[tensor_index]
            assert self.netB_shapes[i] == tensors[tensor_index].shape
            tensor_index += 1

        C_state = OrderedDict()
        for i, k in enumerate(self.netC_keys):
            C_state[k] = tensors[tensor_index]
            assert self.netC_shapes[i] == tensors[tensor_index].shape
            tensor_index += 1

        assert tensor_index == len(tensors)
        return F_state, B_state, C_state


