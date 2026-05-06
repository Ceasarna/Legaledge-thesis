# LegalEdge Thesis Repository

This repository contains the source code, smart-contract components, and figure-generation scripts for the master thesis:

**Design and Evaluation of a Contract-Aware Federated Reinforcement Learning Framework for Smart EV Charging — A LegalEdge Prototype**

Tommy Ekberg
Master's Degree Project in AI and Data Science
Department of Computer and Systems Sciences, Stockholm University

## Repository Purpose

This repository documents the implementation artefact developed for the thesis. The artefact, LegalEdge, investigates how blockchain-based smart contracts can be integrated with Federated Reinforcement Learning (FRL) to support contract-aware smart EV charging.

The repository is intended primarily for code inspection and as a record of the artefact described in the thesis. It contains the main source code, smart-contract code, evaluation and plotting scripts, and selected experiment outputs relevant to the thesis.

## Repository Status

This repository was created from the original development environment used during the thesis work. The original environment contained temporary files, local paths, intermediate logs, and experimental material that were useful during development but not suitable for a clean thesis repository.

Some scripts may require local configuration before execution, including paths to trained model files, generated logs, Nord Pool price data, blockchain artefacts, or evaluation outputs from the original experimental runs. The repository is therefore best understood as a cleaned thesis-code archive rather than a fully automated reproduction package. It is not guaranteed that all experiments can be reproduced immediately after cloning. In case you would like to reproduce the results, contact the author.

## Thesis Context

LegalEdge is a contract-aware Federated Reinforcement Learning prototype for simulated smart EV charging. It combines a simulated EV charging environment, PPO-based reinforcement learning agents, federated orchestration using Flower, blockchain-based smart contracts, and a Python/Web3 bridge between the learning environment and the smart-contract layer.

The main design idea is to place governance outside the learned policy. Charging actions are evaluated by an external smart-contract layer, and the resulting compliance signal is returned to the reinforcement learning loop.

## Repository Structure

```text
Legaledge-thesis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── contracts/                          
│   ├── hardhat.config.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── contracts/                      
│   │   ├── SmartChargingGovernance.sol
│   │   └── CheckpointGovernance.sol
│   └── ignition/
│       └── modules/
│           ├── SmartChargingGovernance.ts
│           └── CheckpointGovernance.ts
│
└── src/
    ├── envs/
    │   ├── charging_env.py
    │   └── helper/
    │       ├── EV.py
    │       ├── price_parser.py
    │       └── individual_training.py
    │
    ├── federated/
    │   ├── client_app.py
    │   ├── server_app.py
    │   ├── task.py
    │   └── config.py
    │
    ├── governance/
    │   └── contract_bridge.py
    │
    └── analysis/
        ├── eval_collective.py
        ├── eval_comparison.py
        ├── eval_stakeholder.py
        ├── eval_throughput.py
        ├── extract_tb_times.py
        ├── plot_checkpoint_collective.py
        ├── plot_individual_governance_training.py
        ├── plot_ray_throughput.py
        ├── sweep.py
        ├── throughput_ray_standalone.py
        └── xai.py
```

## File-Level Overview

### `src/envs/`

Contains the EV charging simulation environment used by the reinforcement learning agents.

- `charging_env.py` — Main smart charging environment. Defines the observation space, action handling, charging/discharging logic, reward calculation, and contract-aware feedback used during training and evaluation.
- `helper/EV.py` — EV-related properties and helper logic for different vehicle models.
- `helper/price_parser.py` — Parses electricity price data used by the charging environment. The thesis experiments rely on Nord Pool electricity price data, which is not bundled in this repository.
- `helper/individual_training.py` — Helper functionality for individual local training runs.

### `src/federated/`

Contains the federated reinforcement learning setup.

- `client_app.py` — Flower client-side logic for local agent training and communication with the federated server.
- `server_app.py` — Flower server-side orchestration for federated training rounds and aggregation.
- `task.py` — Shared training and evaluation task logic used by the federated setup.
- `config.py` — Configuration values for the federated learning setup.

### `src/governance/`

Contains the smart-contract governance layer and the Python/Web3 bridge.

- `SmartChargingGovernance.sol` — Solidity contract for verifying smart charging compliance rules.
- `SmartChargingGovernance.ts` — TypeScript script for deploying or interacting with the smart charging governance contract.
- `CheckpointGovernance.sol` — Solidity contract used for checkpoint-based governance and collective rule verification.
- `CheckpointGovernance.ts` — TypeScript script for deploying or interacting with the checkpoint governance contract.
- `contract_bridge.py` — Python bridge between the reinforcement learning environment and the blockchain smart contracts. Submits charging decisions to the contract layer and returns compliance feedback to the learning loop.
- `config/` — Hardhat and TypeScript configuration files used for contract compilation and deployment.

### `src/analysis/`

Contains evaluation, plotting, and thesis figure/table generation scripts.

- `eval_comparison.py` — Evaluates individual-governance configurations, including no-contract and contract-aware policies.
- `eval_collective.py` — Evaluates collective-governance behaviour under shared capacity constraints.
- `eval_stakeholder.py` — Evaluates stakeholder-oriented trade-offs between cost, state-of-charge fulfilment, and compliance.
- `eval_throughput.py` — Evaluates throughput and blockchain/system overhead from experiment logs.
- `extract_tb_times.py` — Extracts timing information from TensorBoard or training logs.
- `plot_checkpoint_collective.py` — Generates plots for checkpoint-based collective-governance experiments.
- `plot_individual_governance_training.py` — Generates training plots for the individual-governance scenario.
- `plot_ray_throughput.py` — Generates throughput plots from Ray-based scaling experiments.
- `sweep.py` — Runs or processes parameter sweeps for stakeholder and multi-objective evaluation.
- `throughput_ray_standalone.py` — Standalone throughput/stress-test script for operational viability experiments.
- `xai.py` — SHAP-based analysis for explaining selected charging decisions.

## Data Availability

The charging environment uses electricity price data based on the Nord Pool market. The exact data files used during the thesis experiments are not redistributed in this repository. The format expected by `price_parser.py` is documented in that file.

## What Is Not Included

The repository is focused on thesis-relevant source code and artefacts. It excludes temporary development files, local machine configuration, large raw logs, large model checkpoints, generated Hardhat artefacts, virtual environments, cache files, unrelated experiments, and IDE-specific files.

## Author

Tommy Ekberg
Master's Degree Project in AI and Data Science
Department of Computer and Systems Sciences, Stockholm University