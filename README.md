# LegalEdge Thesis Repository

This repository contains the source code, smart-contract components, experiment artefacts, and figure-generation scripts for the master thesis:

**Design and Evaluation of a Contract-Aware Federated Reinforcement Learning Framework for Smart EV Charging**  
**A LegalEdge Prototype**  
Tommy Ekberg  
Master’s Degree Project in AI and Data Science  
Department of Computer and Systems Sciences, Stockholm University

## Repository Purpose

This repository documents the implementation artefact developed for the thesis. The artefact, LegalEdge, investigates how blockchain-based smart contracts can be integrated with Federated Reinforcement Learning (FRL) to support contract-aware smart EV charging.

The repository is intended to make the thesis code easier to inspect. It contains the main source code, smart-contract code, evaluation scripts, generated figures, and selected experiment outputs relevant to the thesis.

## Repository Status

This repository is a cleaned thesis-code archive derived from the original development environment used during the research process. The original environment contained temporary files, local paths, intermediate logs, model artefacts, and experimental clutter.

This repository should therefore be understood primarily as a documented source-code repository for the thesis artefact, not as a fully automated reproduction package. Some scripts may require local configuration, trained models, generated logs, or blockchain artefacts before execution.

Hardcoded local paths from the original development environment should be replaced with repository-relative paths or environment variables before running the scripts on another machine.

## Thesis Context

LegalEdge is a contract-aware Federated Reinforcement Learning prototype for simulated smart EV charging. It combines:

- a simulated EV charging environment,
- PPO-based reinforcement learning agents,
- federated orchestration using Flower,
- blockchain-based smart contracts,
- a Python/Web3 bridge between the learning environment and smart contracts,
- evaluation scripts for compliance, cost, state-of-charge fulfilment, stakeholder trade-offs, explainability, and operational overhead.

The main design idea is to place governance outside the learned policy. Charging actions are evaluated by an external smart-contract layer, and the resulting compliance signal is returned to the reinforcement learning loop.

## Research Question

The thesis addresses the following research question:

> How can a contract-aware Federated Reinforcement Learning framework be realized using blockchain-based smart contracts to enable autonomous governance among heterogeneous smart charging stakeholders?

## Repository Structure

```text
Legaledge-thesis/
├── README.md
├── requirements.txt
├── .gitignore
│
├── contracts/
│   └── Smart contract deployment/build-related files
│
├── experiments/
│   └── Experiment outputs, logs, and intermediate run artifacts
│
├── figures/
│   └── Thesis figures and generated plots
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
    │   ├── SmartChargingGovernance.sol
    │   ├── SmartChargingGovernance.ts
    │   ├── CheckpointGovernance.sol
    │   ├── CheckpointGovernance.ts
    │   ├── contract_bridge.py
    │   └── config/
    │       ├── hardhat.config.ts
    │       ├── package.json
    │       └── tsconfig.json
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