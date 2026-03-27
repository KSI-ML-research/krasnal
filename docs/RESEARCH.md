# Experiment Notes

## 2026-03-27

Tested Muon optimizer - 50% slower in training iterations compared to baseline.
Possible fix: increase batch size or adjust hyperparameters.

## 2026-03-17

Tested architectural variants:

- SwiGLU
- ReLU^2
- QK-norm

No significant improvement was observed over the baseline in the current training and evaluation setup.
