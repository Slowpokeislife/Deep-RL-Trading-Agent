# Experiment log and interpretation

## Design recorded in the academic report (13 October 2025)

- Financial dataset: Yandex (YNDX) historical quotes from 2015 and 2016.
- Original code defaults: train on 2016, validation on 2015.
- Configurations compared: last 10 bars vs. last 3 bars.
- Training duration described as approximately 12 hours per configuration.
- Checkpoints cited in the report: `mean_value-0.367.data` (10 bars) and `mean_value-0.124.data` (3 bars).
- Charts include a baseline commission assumption and additional scenarios, including 0%, 0.05%, 0.06% and 0.075%; see `Resultados/`.

## How to read the plots

The vertical axis in `run_model.py` is the running sum of the environment's reward values, labeled `Reward, %`. These are simulation reward curves and not a verified brokerage-equity curve. Changes in commissions should be interpreted as sensitivity to assumptions, rather than proof of future returns.

The original report describes weaker validation behavior in certain settings and changes in results with smaller commissions. It does not present a standardized numerical benchmark table, confidence intervals, or out-of-time forward test.

## Important evaluation considerations

1. **Chronology:** training on 2016 and validating on 2015 does not simulate predicting a future period; reverse the years or use a later unseen holdout for stronger evaluation.
2. **Randomness:** `run_model.py` uses epsilon 0.02 and does not set a seed, so results can vary across runs.
3. **Comparison:** record identical training budgets, fee assumptions and evaluation slices for the 3-bar and 10-bar configurations.
4. **Selection:** a checkpoint named `mean_value-...` is selected by a mean predicted Q-value rather than necessarily by held-out profit.
5. **Costs:** the environment includes a simplified percentage commission, not detailed fills, liquidity, spread, slippage or market impact.
6. **Benchmarks:** compare against buy-and-hold, no-trade and random policies before arguing the RL strategy improves returns.

## Suggested future result table

| Model | Bars | Test period | Commission | Mean episode reward | Profit per trade | Max drawdown | Baseline comparison |
|---|---:|---|---:|---:|---:|---:|---|
| Dueling DQN (feed-forward) | 10 | TBD | TBD | TBD | TBD | TBD | TBD |
| Dueling DQN (feed-forward) | 3 | TBD | TBD | TBD | TBD | TBD | TBD |

`TBD` deliberately indicates missing measurements rather than invented statistics.
