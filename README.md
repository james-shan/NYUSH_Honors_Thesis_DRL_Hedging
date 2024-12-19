# Optimal Hedging via Deep Reinforcement Learning with Soft Actor-Critic

This repository contains the implementation of the research project **"Optimal Hedging via Deep Reinforcement Learning with Soft Actor-Critic"**, conducted as part of an honors thesis in the Business and Economics Honors Program at NYU Shanghai. The project explores how deep reinforcement learning (DRL) can optimize hedging strategies, using the Soft Actor-Critic (SAC) algorithm to minimize trading costs and variance.

---

## Overview

Hedging is a key aspect of financial risk management, allowing practitioners to mitigate potential losses. Traditional hedging methods, such as Delta hedging, often face limitations due to high trading costs and assumptions of constant volatility. This project addresses these challenges by leveraging the SAC algorithm to design more adaptive and efficient hedging strategies. 

The study evaluates SAC’s performance under both **Geometric Brownian Motion (GBM)** and **Stochastic Volatility** stock price models, comparing it with the Delta hedging strategy and the Deep Deterministic Policy Gradient (DDPG) algorithm.

---

## Methodology

### Problem Setting
The hedging environment is designed to replicate real-world scenarios:
- **State Space**: The environment includes stock price, time to maturity, current holdings, and the option price. Notably, the model does not explicitly use “Greeks,” as it allows the RL agent to infer them from training data.
- **Action Space**: Represents the number of shares to hold after trading, normalized between -1 and 1 for ease of training.
- **Reward Function**: Balances variance and transaction costs using a quadratic utility framework, encouraging the agent to minimize both simultaneously.

### Algorithms
- **Soft Actor-Critic (SAC)**: An off-policy, maximum-entropy algorithm that enhances exploration and prevents premature convergence. SAC uses two Q-networks to reduce overestimation bias and introduces a stochastic actor for robust decision-making.
- **Delta Hedging**: Traditional benchmark strategy, designed to maintain a Delta-neutral position.
- **Deep Deterministic Policy Gradient (DDPG)**: Another reinforcement learning algorithm employed as a baseline for comparison.

### Training Environment
- Built using **Gymnasium** in Python.
- Simulations conducted with parameters:
  - Volatility: 0.15 (annual).
  - Trading frequency: 5 times per day.
  - Call option maturity: 10 or 30 days.
  - Risk-aversion parameter (\( \kappa \)): 0.1.
  - Transaction cost parameter: 0.003.

---

## Results

### Performance Metrics
The strategies were evaluated on:
1. **Profit and Loss (P&L)**: A measure of financial performance.
2. **Trading Costs**: Total costs incurred for hedging.
3. **Rewards**: A combined metric of cost and variance reduction.

### Key Findings
1. **Geometric Brownian Motion (GBM) Stock Prices**:
   - SAC outperformed Delta hedging and DDPG, achieving lower costs and more stable P&L distributions.
   - SAC managed to reduce trading costs by approximately 50% compared to Delta hedging while maintaining comparable risk management.

2. **Stochastic Volatility Stock Prices**:
   - SAC demonstrated robustness under more volatile conditions, though its performance margin over Delta hedging and DDPG narrowed.
   - SAC and DDPG exhibited similar trading costs, but SAC achieved better balance between variance and cost.

### Summary of Results
| Method          | Mean P&L | Std P&L | Mean Cost | Std Cost | Mean Reward | Std Reward |
|------------------|----------|---------|-----------|----------|-------------|------------|
| **SAC (GBM)**   | -48.61   | 34.88   | 48.23     | 12.63    | -98.94      | 65.25      |
| **DDPG (GBM)**  | -43.70   | 40.90   | 42.97     | 8.59     | -119.16     | 67.13      |
| **Delta (GBM)** | -96.29   | 33.55   | 96.44     | 28.21    | -146.30     | 55.43      |

Detailed visualizations, including Kernel Density Estimates (KDE) of rewards and trading costs, are provided in the `results/` folder.

---

## Future Work

The study highlights several potential areas for improvement:
1. **Bayesian Soft Actor-Critic**: Introducing Bayesian methods to handle uncertainty and improve model robustness, especially in real-world scenarios.
2. **Curriculum Learning**: Progressively training the agent from simple to complex scenarios to enhance generalization and stability.
3. **Real-World Data**: Extending the experiments to real market datasets to improve applicability.
4. **Multi-Step Learning**: Incorporating multi-step returns to improve sample efficiency and reduce data requirements.

---

## Citation

If you use this work in your research, please cite it as follows:

```bibtex
@thesis{shan2024hedging,
  title={Optimal Hedging via Deep Reinforcement Learning with Soft Actor-Critic},
  author={Zhihao Shan},
  year={2024},
  school={NYU Shanghai},
  type={Honors Thesis}
}
