# Making Training Work

Diagnosing and fixing a network that would not train, one change at a time.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Run
python Lab02_diagnoses.py # reproduces the failure and reports the gradient check
python Lab02_changes.py # runs each fix stage and saves a comparison plot

## Diagnosis
We checked why BrokenNet would not train by inspecting neuron activations in the forward pass and gradient magnitudes after one backward pass.

In BrokenNet, every layer has bias = -2.0. This negative bias shifts all inputs into negative numbers before passing them through ReLU. Because ReLU turns any negative number into 0.0, 100% of the neurons output 0.0.

During backpropagation, the slope of ReLU at 0 is 0. Multiplying by 0 along the chain rule completely wipes out the gradients across all hidden layers (mean gradient at Layer 0 is 0.0). The optimizer needs positive number to updates weights.

# Stages and results
Unfixed: Final Loss = 0.6893
Fix1: Final Loss = 0.4015
Fix2: Final Loss = 0.1397
Fix3: Final Loss = 0.0005

# Conclusion
Stage 1 (fixing initialization) made the single biggest difference.

When gradients are zero, no optimizer can update any weights. By switching to Kaiming normal initialization and setting the bias to 0.0, the inputs were centered around zero instead of negative values. This kept about 50% of the neurons active, brought the gradients back to life, and allowed the loss to drop below the random guessing baseline of 0.4015.

Stage 2 (adding BatchNorm1d) helped further by keeping the scale of activations stable across all 8 layers. Finally, Stage 3 (Adam) fine-tuned the learning rate to reach a final loss of 0.005.

