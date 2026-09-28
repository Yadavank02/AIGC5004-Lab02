import torch
import torch.nn as nn


class BrokenNet(nn.Module):
    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__()
        layers = []
        n_in = in_features
        for _ in range(depth):
            layers.append(nn.Linear(n_in, width))
            n_in = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)

        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0.0, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)


def make_toy_classification(n=256, in_features=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_features, generator=g)
    true_w = torch.randn(in_features, 1, generator=g)
    logits = x @ true_w
    y = (logits > 0).float().squeeze(-1)
    return x, y


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BrokenNet()
    x, y = make_toy_classification(seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("Training BrokenNet for 50 epochs with plain SGD...")
    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 49:
            print(f"epoch {epoch:3d}   loss = {loss.item():.4f}")