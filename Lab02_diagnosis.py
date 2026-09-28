import torch
import torch.nn as nn
from Lab02_starter import BrokenNet, make_toy_classification

def reproduce_failure():
    print("1: Reproducing the BrokenNet Training Failure")
   
    torch.manual_seed(0)
    model = BrokenNet()
    x, y = make_toy_classification(seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 49:
            print(f"epoch {epoch:3d}   loss = {loss.item():.4f}")

    print("Untrained random guess loss on binary BCE is ln(2) ≈ 0.6931.")
    print(f"Final epoch loss: {loss.item():.4f} (Network fails to train).")

def per_layer_grad(model):
    magnitudes = []
    for layer in model.hidden:
        if layer.weight.grad is not None:
            magnitudes.append(layer.weight.grad.abs().mean().item())
        else:
            magnitudes.append(0.0)
    return magnitudes

def run_diagnosis():

    print("2: Forward Pass Activation Check (Layer-by-Layer)")
    torch.manual_seed(0)
    model = BrokenNet()
    x, y = make_toy_classification(seed=0)

    # Track layer-by-layer activations in the forward pass
    current_input = x
    for idx, layer in enumerate(model.hidden):
        pre_act = layer(current_input)
        post_act = torch.relu(pre_act)

        # Percentage of neurons in this layer with output exactly equal to 0.0
        dead_ratio = (post_act == 0.0).float().mean().item() * 100.0
        mean_pre_val = pre_act.mean().item()

        print(
            f"Layer {idx} | Mean Pre-activation: {mean_pre_val:.2f} | "
            f"Dead Neurons (Output == 0.0): {dead_ratio:.2f}%"
        )
        current_input = post_act
    print("3: Backward Pass Gradient Check (Layer-by-Layer)")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    preds = model(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
    loss.backward()

    # Read per-layer gradient magnitudes
    grad_means = per_layer_grad(model)

    print(f"{'Layer':>10} | {'Mean |gradient|':>20} | {'Weight L2 Norm':>20}")

    for i, g in enumerate(grad_means):
        l2_norm = model.hidden[i].weight.grad.norm().item()
        print(f"Layer {i} | {g:.2e} | {l2_norm:.2e}")

    out_mean = model.out.weight.grad.abs().mean().item()
    out_l2 = model.out.weight.grad.norm().item()
    print(f"{'Output'} | {out_mean:.2e} | {out_l2:.2e}")

    print("DIAGNOSTIC SUMMARY:")
    layer0_grad = grad_means[0]
    layer7_grad = grad_means[-1]

    print(f"Layer 0 Mean |gradient|: {layer0_grad:.2e}")
    print(f"Layer 7 Mean |gradient|: {layer7_grad:.2e}")

    if layer0_grad == 0.0:
        print("\n--- Diagnostic Verdict ---")
        print("Diagnosis: Dying ReLU / Vanishing Gradient.")
        print("Layer 0 weight gradient is exactly 0.00e+00.")
        print("\nWhy it failed:")
        print("1. Every layer has bias = -2.0, which forces all pre-activations negative.")
        print("2. ReLU turns all negative inputs into 0.0, killing 100% of neurons.")
        print("3. Because the slope of ReLU at 0 is 0, backprop multiplies by 0.")
        print("4. This wipes out the gradients, so the optimizer cannot update any weights.")
if __name__ == "__main__":
    reproduce_failure()
    run_diagnosis()