import theseus as th
import torch
import torch.nn as nn


# simple example from theseus discussion forum at github
# https://github.com/facebookresearch/theseus/discussions/285
# important notes:
# - NN has nested Theseus layer
# - model called insode the objective -> cost function. thus it si fixed during inner loop stage

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # some arbitrary NN
        self.nn = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1)) # 

        # Add a theseus layer with a single cost function whose error depends on the NN
        objective = th.Objective()
        x = th.Vector(2, name="x")
        y = th.Vector(1, name="y")
        # This cost function computes `err(x) = nn(x) - y`
        objective.add(th.AutoDiffCostFunction([x], self._error_fn, 1, aux_vars=[y]))
        optimizer = th.LevenbergMarquardt(objective, max_iterations=10)
        self.layer = th.TheseusLayer(optimizer)

    def _error_fn(self, optim_vars, aux_vars):
        x = optim_vars[0].tensor
        y = aux_vars[0].tensor
        err = self.nn(x) - y
        return err

    # Run theseus so that NN(x*) is close to y
    def forward(self, y):
        x0 = torch.ones(y.shape[0], 2)
        sol, info = self.layer.forward(
            {"x": x0, "y": y}, optimizer_kwargs={"damping": 0.1}
        )
        print("Optim error: ", info.last_err.item())
        return sol["x"]


# Outer loop will modify NN weights to make x* as small as possible, while
# inner loop guarantees that NN(x*) is close to y
m = Model()
optim = torch.optim.Adam(m.nn.parameters(), lr=0.01)
y = torch.ones(1, 1)
for i in range(5):
    optim.zero_grad()
    xopt = m.forward(y)
    loss = (xopt**2).sum()
    loss.backward()
    optim.step()
    print("Outer loss:", loss.item(), "\n------------------------")
