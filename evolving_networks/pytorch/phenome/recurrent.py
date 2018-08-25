import torch

from evolving_networks.pytorch.phenome.phenome import Phenome


class RecurrentNetwork(Phenome):
    def __init__(self, genome, config):
        super(RecurrentNetwork, self).__init__()
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
        inputs = inputs.view(1, 1, self.model.rnn.input_size)
        with torch.no_grad():
            output = self.model(inputs)
        return output.numpy().tolist()[0][0]

    def reset(self, hard=False):
        self.model.reset()
