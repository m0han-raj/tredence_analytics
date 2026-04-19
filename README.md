# Report on Self-Pruning Neural Network on CIFAR-10

## Overview

This report covers my implementation of a self-pruning feed-forward network
for CIFAR-10. Instead of pruning the model after training, the network
learns during training which of its own weights to switch off by attaching a
learnable gate to every weight.

## Approach

Every weight `w` in each linear layer is paired with a second learnable
parameter `s`, called a gate score. In the forward pass, the weight is
multiplied by `sigmoid(s)` before use:

```
effective_weight = w * sigmoid(s)
output           = effective_weight @ input + bias
```

Both `w` and `s` are registered as `nn.Parameter`, so the optimizer updates
them together during `loss.backward()`.

A custom `PrunableLinear` class replaces the standard linear layer. Its
forward pass is `x @ (weight * gate).t() + bias`. No `torch.nn.Linear` or
`F.linear` is used anywhere.

## Why L1 on sigmoid gates creates sparsity

The loss combines classification with an L1 penalty on all gates:

```
Loss = CrossEntropy(logits, y) + λ · Σ sigmoid(s)
```

Since `sigmoid(s)` is always positive, the sum is exactly the L1 norm of
the gate vector. This creates two competing forces on every gate:

1. The **L1 penalty** pulls every gate toward 0, whether or not the weight
   behind it is useful.
2. **Cross-entropy** pulls a gate up only if closing it would hurt the
   prediction.

Gates attached to useful weights win this tug of war and stay near 1. Gates
attached to redundant weights have nothing defending them, so the L1
pressure drives them to 0. The final distribution ends up bimodal: a large
cluster at 0 and a smaller cluster near 1.

## Hyperparameters

| Hyperparameter               | Value                                  |
|:-----------------------------|:---------------------------------------|
| Network architecture         | 3072 → 512 → 256 → 10 (PrunableLinear) |
| Activation                   | ReLU (between hidden layers)           |
| Weight initialisation        | Kaiming uniform                        |
| Gate score initialisation    | 0.0 (every gate starts at 0.5)         |
| Optimizer                    | Adam                                   |
| Learning rate (weights/bias) | 1e-3                                   |
| Learning rate (gate scores)  | 5e-3                                   |
| Batch size                   | 256                                    |
| Epochs                       | 20                                     |
| Loss function                | CrossEntropy + λ · Σ sigmoid(s)        |
| Lambda (λ) values swept      | 1e-4, 1e-3, 1e-2                       |
| Sparsity threshold           | 1e-2                                   |

Gate scores use a higher learning rate than weights because they need to
travel further in parameter space. With a shared lr=1e-3 the gates
saturated around 0.05 due to the vanishing sigmoid gradient near zero.
Splitting into two parameter groups solved this without destabilising
weight training.

## Results

Sparsity level is the percentage of gates with value below `1e-2`.

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:-------------:|:--------------:|
| 1e-4       | **53.71%**    | 96.32%         |
| 1e-3       | 49.27%        | 99.84%         |
| 1e-2       | 37.90%        | 99.99%         |

### Interpretation

- **λ = 1e-4 is the best trade-off.** The network prunes 96% of its weights
  but only loses a few points of accuracy. Effectively, only about 4% of
  the original connections are doing real work.
- **λ = 1e-3 over-prunes.** Sparsity improves only slightly (from 96% to
  99.84%) but accuracy drops by more than 4 points. Past this point the
  extra compression costs more than it is worth.
- **λ = 1e-2 collapses the network.** Almost every gate closes. Accuracy is
  still well above the 10% random baseline, so a tiny number of surviving
  gates still carry signal, but the network has lost most of its capacity.

The pattern is clean: stronger sparsity pressure leads to more pruning but
lower accuracy, and the best operating point is at λ = 1e-4.

### Training observation

In all three runs, sparsity stays at 0% for roughly the first 11 epochs
even though the gates are clearly dropping. They just have not crossed the
strict `1e-2` threshold yet. Once they do, sparsity jumps from 0% to 90%+
within two or three epochs. Higher λ simply makes this breakthrough happen
faster.

## Gate value distribution

See `gate_distribution.png`, generated for the best model (λ = 1e-4). The
y-axis is on log scale because the spike at 0 dwarfs everything else. The
bimodal shape is clearly visible: a tall spike at `g ≈ 0` containing about
96% of gates, and a smaller cluster of surviving gates spread across higher
values.
