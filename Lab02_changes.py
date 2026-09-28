import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Lab02_starter import BrokenNet, make_toy_classification

class Unfixed(nn.Module):
    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__()
        layers = []
        n_in = in_features
        for _ in range(depth):
            layers.append(nn.Linear(n_in, width))
            n_in = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)

        # Defective initialization causing the training failure
        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0.0, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)

class Fix1(nn.Module):
    # Weights Kaiming Normal, biases zeroed
    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__()
        layers = []
        n_in = in_features
        for _ in range(depth):
            layers.append(nn.Linear(n_in, width))
            n_in = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)

        # Apply Kaiming normal initialization specifically designed for ReLU
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        # Output layer initialized with Xavier normal
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)

class Fix2(nn.Module):
    # Weights Kaiming Normal, biases zeroed
    #Adds normalization (BatchNorm1d)
    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__()
        linears =[]
        norms = []
        n_in = in_features
        for _ in range(depth):
            linears.append(nn.Linear(n_in, width))
            norms.append(nn.BatchNorm1d(width))
            n_in = width
        self.linears = nn.ModuleList(linears)
        self.norms = nn.ModuleList(norms)
        self.out = nn.Linear(width, 1)

        # Apply Kaiming normal initialization specifically designed for ReLU
        for layer in self.linears:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for linear, norm in zip(self.linears, self.norms):
            x = torch.relu(norm(linear(x)))
        return self.out(x)

def train_model(model_class, optimizer_mode="sgd", lr=0.1, epochs=50, seed=0):

    torch.manual_seed(0)
    model = model_class()
    x, y = make_toy_classification(seed=seed)
    if optimizer_mode == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_mode == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer mode. Choose 'sgd' or 'adam'.")

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}   loss = {loss.item():.4f}")
    return losses

def main():
    stages_results = {}
        #print("Training BrokenNet for 50 epochs with plain SGD...")
        #stages_results["BrokenNet"] = train_model(BrokenNet, optimizer_mode="sgd", lr=0.1, epochs=50, seed=0)
    print("\nTraining Unfixed (BorkenNet) for 50 epochs with plain SGD...")
    stages_results["Unfixed"] = train_model(Unfixed, optimizer_mode="sgd", lr=0.1, epochs=50, seed=0)
    print("\nTraining Fix1 for 50 epochs with plain SGD...")
    stages_results["Fix1"] = train_model(Fix1, optimizer_mode="sgd", lr=0.1, epochs=50, seed=0)
    print("\nTraining Fix2 for 50 epochs with plain SGD...")
    stages_results["Fix2"] = train_model(Fix2, optimizer_mode="sgd", lr=0.1, epochs=50, seed=0)     
    print("Training Fix2 for 50 epochs with optimizer...")
    stages_results["Fix3"] = train_model(Fix2, optimizer_mode="adam", lr=0.01, epochs=50, seed=0)

    print("Stages and Final Losses")
    for stage, losses in stages_results.items():
        print(f"{stage}: Final Loss = {losses[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    for stage, losses in stages_results.items():
        plt.plot(losses, label=stage)
    plt.title("Training Loss Across Different Stages")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.savefig("comparison_plot.png", dpi=300)
    print("\nComparison plot successfully saved to 'comparison_plot.png'.") 
if __name__ == "__main__":
    main()