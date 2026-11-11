# Reproducibility and publication checklist

## Original source audit

- `train_model.py` currently has `BARS_COUNT = 10`. To reproduce the reported 3-bar run, set this constant to 3 before training.
- `lib/environ.py` defaults to 10 bars, but the training script passes its own bar setting explicitly.
- `train_model.py -r modelo_10bar` saves to `saves/simple-modelo_10bar` (the `simple-` prefix is added by code).
- Training uses a very long / effectively open-ended loop: stop it manually when appropriate and retain useful checkpoints.
- `run_model.py` saves an output image in the current directory, not in `Resultados/` automatically.
- Some checkpoints may exceed practical GitHub repository size limits. Prefer one representative checkpoint per model in GitHub Releases.
- `data/` files are included in the uploaded archive, but license/provenance for redistributing historical quotes should be checked.

## Prior to a reproducibility claim

1. Capture `python --version`, `pip freeze`, OS and GPU information.
2. Pin tested versions (Torch, PTAN, Ignite and Gymnasium); do not assume unpinned newest versions work together.
3. Add a `--bars` CLI option to training for an explicit, reproducible configuration.
4. Seed NumPy, Torch, Gymnasium and the evaluation's epsilon-greedy randomness.
5. Add a fixed number of training steps and model-selection criteria.
6. Evaluate on a later untouched calendar period and log comparison baselines.
7. Verify `python -m pytest test/` and reproduce one plot on a clean clone.

## Git history hygiene

Because the uploaded ZIP contains `.git/` and more than 100 saved checkpoints, adding a `.gitignore` does **not** remove already tracked artifacts or past Git objects. To stop tracking model/log folders going forward while preserving local files:

```bash
git rm -r --cached saves runs
git add README.md requirements.txt .gitignore docs/
git commit -m "docs: document DQN trading experiments"
git push origin main
```

Run the `git rm` command only if you want those files removed from the remote repository in the next commit. If large binary objects remain in Git history, normal removal does not shrink that history; rewriting it is a separate, potentially disruptive task.

## Attribution and rights

This is an academic reproduction/adaptation of the stock-trading exercise in Maxim Lapan's _Deep Reinforcement Learning Hands-On_. State clearly which parts came from the reference and which experiments and interpretation are original. Confirm the correct upstream license and dataset distribution rights; do not automatically add an MIT license to all code without checking.
