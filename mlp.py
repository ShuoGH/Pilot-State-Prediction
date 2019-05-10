import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNetwork(nn.Module):
    '''
    Parameters:
        n_features: the input features number.
        n_neurons: [x1,x2] two hidden layers and the number of neurons in each layer.
    Return:
        softmax probability.
    '''

    def __init__(self, n_features, n_neurons, dropouts):
        super().__init__()
        self.layer1 = nn.Linear(in_features=n_features,
                                out_features=n_neurons[0])
        self.dropout1 = nn.Dropout(dropouts[0])
        self.layer2 = nn.Linear(
            in_features=n_neurons[0], out_features=n_neurons[1])
        self.dropout2 = nn.Dropout(dropouts[1])
        self.out_layer = nn.Linear(in_features=n_neurons[1], out_features=4)

    def forward(self, X):
        out = F.relu(self.layer1(X))
        out = self.dropout1(out)
        out = F.relu(self.layer2(out))
        out = self.dropout2(out)
        out = self.out_layer(out)
        return F.log_softmax(out, dim=1)


# Note:
# The loss used in NLLLoss, can have weights
# It's very useful when you train the unbalanced data set.
#
# Maybe I can try to use pytorch to ensemble adaboosting?
# See the doc: https://pytorch.org/docs/stable/nn.html?highlight=nllloss#torch.nn.NLLLoss
