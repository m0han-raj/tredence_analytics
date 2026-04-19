import torch
import torch.nn.functional as F
from network import Net
from data import train_loader , test_loader , device
from sparser import sparsity_loss , sparsity_pct , all_gates
import matplotlib.pyplot as plt

def train(lam, epochs=20):
    model = Net().to(device)
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params  = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    opt = torch.optim.Adam([
        {"params": other_params, "lr": 1e-3},
        {"params": gate_params,  "lr": 5e-3},
    ])
 
    for e in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y) + lam * sparsity_loss(model)
            opt.zero_grad(); loss.backward(); opt.step()
        g = all_gates(model).detach()
        print(f"  epoch {e + 1}/{epochs}  gate_mean={g.mean():.3f}  "
              f"<1e-1={(g < 1e-1).float().mean() * 100:.1f}%  "
              f"<1e-2={(g < 1e-2).float().mean() * 100:.1f}%")
 
    # test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total * 100, sparsity_pct(model), all_gates(model).detach().cpu().numpy()

results = []
for lam in [1e-4, 1e-3, 1e-2]:
    print(f"\nTraining with lambda = {lam}")
    acc, sp, gates = train(lam)
    results.append((lam, acc, sp, gates))
 
print("\n| Lambda | Test Accuracy | Sparsity |")
print("|--------|---------------|----------|")
for lam, acc, sp, _ in results:
    print(f"| {lam:.0e} | {acc:.2f}% | {sp:.2f}% |")
 
# plot gates for the best pruned model
best = max([r for r in results if r[2] > 50] or results, key=lambda r: r[1])
plt.hist(best[3], bins=100); plt.yscale("log")
plt.xlabel("Gate value"); plt.ylabel("Count")
plt.title(f"Final gate distribution (lambda = {best[0]:.0e})")
plt.savefig("gate_distribution.png")
print(f"\nSaved gate_distribution.png for lambda = {best[0]:.0e}")