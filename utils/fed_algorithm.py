from collections import OrderedDict

class FederatedAlgorithm:
    def __init__(self, train_sizes, init_model):
        self.train_sizes = train_sizes
        if type(init_model) == OrderedDict:
            self.param_keys = init_model.keys()
        else:
            self.param_keys = init_model.cpu().state_dict().keys()

    def update(self, local_models, client_indices, global_model=None):
        pass

class FedAvg(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)

    def update(self, local_models, client_indices, global_model=None):
        num_training_data = sum([self.train_sizes[idx] for idx in client_indices])
        update_model = OrderedDict()
        for idx in range(len(client_indices)):
            local_model = local_models[idx].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[idx]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if idx == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]
        return update_model