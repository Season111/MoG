import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
import numpy as np

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=300, add_bias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize Fourier coefficients
        self.fourier_coeffs = nn.Parameter(
            torch.randn(2, output_dim, input_dim, grid_size) /
            (np.sqrt(input_dim) * np.sqrt(grid_size))
        )
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def reset_parameters(self):
        self.fourier_coeffs.data.normal_(0, 1 / np.sqrt(self.input_dim * self.grid_size))
        if self.add_bias:
            self.bias.data.zero_()

    def forward(self, x):
        x_shape = x.shape
        out_shape = x_shape[:-1] + (self.output_dim,)
        x = x.view(-1, self.input_dim)

        # Compute Fourier features
        k = torch.arange(1, self.grid_size + 1, device=x.device).view(1, 1, 1, self.grid_size)
        x_reshaped = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * x_reshaped)
        s = torch.sin(k * x_reshaped)
        c = c.view(1, x.shape[0], x.shape[1], self.grid_size)
        s = s.view(1, x.shape[0], x.shape[1], self.grid_size)

        # Compute output
        y = torch.einsum("dbik,djik->bj", torch.cat([c, s], dim=0), self.fourier_coeffs)
        if self.add_bias:
            y += self.bias
        y = y.view(out_shape)
        return y

class GKAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, grid_size, num_layers, dropout, use_bias=False):
        super(GKAN, self).__init__()
        self.dropout = dropout

        # Create input and output linear layers
        self.lin_in = nn.Linear(in_channels, hidden_channels, bias=use_bias)
        self.lin_out = nn.Linear(hidden_channels, out_channels, bias=False)

        # Create KAN layers
        self.kan_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.kan_layers.append(NaiveFourierKANLayer(hidden_channels, hidden_channels, grid_size, add_bias=use_bias))

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        for layer in self.kan_layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        x = self.lin_in(x)
        for layer in self.kan_layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = self.lin_out(x)
        return F.log_softmax(x_final, dim=-1)

class Args:
    def __init__(self):
        self.path = './data/'
        self.name = 'Flickr'
        self.logger_path = 'logger/esm'
        self.dropout = 0.0
        self.hidden_size = 64  # Reduced size
        self.grid_size = 100  # Reduced size
        self.n_layers = 1  # Reduced number of layers
        self.epochs = 1000
        self.early_stopping = 100
        self.seed = 42
        self.lr = 5e-4

args = Args()

# Load Flickr dataset
path = './data/Flickr'
dataset = Flickr(path, transform=NormalizeFeatures())
data = dataset[0]

# Set random splits
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_disassortative_splits(labels, num_classes, trn_percent=0.6, val_percent=0.2):
    labels = labels.cpu()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    percls_trn = int(round(trn_percent * (labels.size()[0] / num_classes)))
    val_lb = int(round(val_percent * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask, val_mask, test_mask

num_classes = dataset.num_classes
train_mask, val_mask, test_mask = random_disassortative_splits(data.y, num_classes)

# Model parameters
in_channels = dataset.num_features
hidden_channels = args.hidden_size
out_channels = num_classes
grid_size = args.grid_size
num_layers = args.n_layers
dropout = args.dropout

# Initialize model and optimizer
model = GKAN(in_channels, hidden_channels, out_channels, grid_size, num_layers, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training and evaluation functions
def train(args, feat, adj, label, mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(feat, adj)
    pred, true = out[mask], label[mask]
    loss = F.nll_loss(pred, true)
    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()

@torch.no_grad()
def eval(args, feat, adj, model):
    model.eval()
    with torch.no_grad():
        pred = model(feat, adj)
    pred = pred.argmax(dim=-1)
    return pred

# Training loop
for epoch in range(args.epochs):
    acc, loss = train(args, data.x, data.edge_index, data.y, train_mask, model, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

# Evaluation
model.eval()
pred = eval(args, data.x, data.edge_index, model)
test_acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
print(f'Test Accuracy: {test_acc:.4f}')
