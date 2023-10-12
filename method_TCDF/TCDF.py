import numpy as np
import pandas as pd
import torch
from depthwise import DepthwiseNet


class ADDSTCN(torch.nn.Module):
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()
        self.target = target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = torch.nn.Conv1d(input_size, 1, 1)
        self._attention = torch.ones(input_size, 1)
        self._attention = torch.autograd.Variable(self._attention, requires_grad=False)
        self.fs_attention = torch.nn.Parameter(self._attention.data)
        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()

    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y1 = self.dwn(x * torch.nn.functional.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1)
        return y1.transpose(1, 2)


def prepare_data(data, target):
    if isinstance(data, pd.DataFrame):
        data = data.values
    x = data.copy()
    y = data[:, [target]]
    y_shift = np.vstack([np.zeros(shape=(1, y.shape[1])), y[:-1, :]])
    x[:, [target]] = y_shift
    y = torch.autograd.Variable(torch.from_numpy(y.astype('float32').T))
    x = torch.autograd.Variable(torch.from_numpy(x.astype('float32').T))
    return x, y

def train(x, y, model_name, optimizer):
    """
    Trains model by performing one epoch and returns attention scores and loss.
    :param x:
    :param y:
    :param model_name:
    :param optimizer:
    :return:
    """
    model_name.train()
    x, y = x[0:1], y[0:1]
    optimizer.zero_grad()
    output = model_name(x)
    attention_score = model_name.fs_attention
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    return attention_score.data, loss


def find_causes(x, target_idx, seed, cuda=False, epochs=1000, max_lag=3, layers=1, learning_rate=0.01,
                optimizer='Adam', dilation_c=4, significance=0.8):
    """
    Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers
    the corresponding time delays.
    :param x: [pandas.DataFrame] All time series (including the dependent), where each column is a time series.
    :param target_idx: The column index of the dependent time series.
    :param seed: Random seed.
    :param cuda: Default False. Whether to use CUDA (GPU).
    :param epochs: Default 1000. Number of epochs in training process.
    :param max_lag: Default 3. Maximum delay to be found
    :param layers: Default 1. Number of layers in the depth-wise convolution neural network, including the output layer,
      excluding input layer.
    :param learning_rate: Default 0.01.
    :param optimizer: Default 'Adam'. Chosen from ('Adam', 'RMSprop').
    :param dilation_c: Default 4. Dilation coefficient, recommended to be equal to 'max_lag + 1'.
    :param significance: Default 0.8. Significance number stating when an increase in loss is significant enough to
      label a potential cause as true (validated) cause.
    :return:
    """
    assert isinstance(epochs, int) and epochs > 0
    assert isinstance(layers, int) and layers > 0
    assert isinstance(max_lag, int) and max_lag >= 0
    assert isinstance(learning_rate, float) and learning_rate > 0
    assert optimizer in ('Adam', 'RMSprop')
    assert isinstance(seed, int) and seed > 0
    assert isinstance(dilation_c, int) and dilation_c > 0

    torch.manual_seed(seed)
    x_train, y_train = prepare_data(x, target_idx)
    x_train = x_train.unsqueeze(0).contiguous()
    y_train = y_train.unsqueeze(2).contiguous()
    input_channels = x_train.size()[1]
    model = ADDSTCN(target=target_idx, input_size=input_channels, num_levels=layers, kernel_size=max_lag + 1,
                    cuda=cuda, dilation_c=dilation_c)
    if cuda:
        model.cuda()
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    scores = None
    first_loss = None
    real_loss = None
    for epoch in range(epochs):
        scores, loss = train(x_train, y_train, model, optimizer)
        if epoch == 0:
            first_loss = loss.cpu().data.item()
        if epoch == epochs - 1:
            real_loss = loss.cpu().data.item()
    scores = scores.view(-1).cpu().detach().numpy()
    sorted_idx = np.argsort(-scores)
    s = scores[sorted_idx]
    # attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
    if s.shape[0] <= 5:
        validated = [i for i in sorted_idx if scores[i] > 1]
    else:
        gaps = - np.diff(s)[s[:-1] >= 1]
        gaps = np.sort(gaps)[::-1]
        # gap should be in first half
        # gap should have index > 0, except if second score < 1
        validated = sorted_idx[:(gaps.shape[0] + 1) // 2].tolist()

    # Apply PIVM (permutes the values) to check if potential cause is true cause
    for idx in validated:
        rng = np.random.RandomState(seed=seed)
        x_train_r = x_train.clone().cpu().numpy()
        rng.shuffle(x_train_r[:, idx, :][0])
        shuffled = torch.from_numpy(x_train_r)
        if cuda:
            shuffled = shuffled.cuda()
        model.eval()
        output = model(shuffled)
        test_loss = torch.nn.functional.mse_loss(output, y_train)
        test_loss = test_loss.cpu().data.item()
        diff = first_loss - real_loss
        test_diff = first_loss - test_loss
        if test_diff > (diff * significance):
            validated.remove(idx)
    return validated, real_loss, s
