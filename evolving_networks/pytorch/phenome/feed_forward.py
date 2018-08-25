import torch

from evolving_networks.pytorch.phenome.phenome import Phenome


class FeedForwardNetwork(Phenome):
    def __init__(self, genome, config):
        super(FeedForwardNetwork, self).__init__()
        self.genome = genome
        self.config = config

        self.model = None

        # Flag for genome removal
        self.is_damaged = False

    def initialize(self, model_class):
        self.model = model_class()
        for param, weights in zip(self.model.parameters(), self.genome.structured_weights):
            param.data = torch.Tensor(weights)

    def activate(self, inputs):
        inputs = torch.Tensor(inputs)
        with torch.no_grad():
            output = self.model(inputs)
        return output.numpy().tolist()

    def reset(self, hard=False):
        pass
