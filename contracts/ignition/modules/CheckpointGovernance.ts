import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

// 100 kWh total capacity across all agents, scaled by 1000
const MAX_CAPACITY_SCALED = 100_000n;
const NUM_AGENTS = 4n;

const CheckpointGovernanceModule = buildModule(
  "CheckpointGovernanceModule",
  (m) => {
    const maxCapacityScaled = m.getParameter("maxCapacityScaled", MAX_CAPACITY_SCALED);
    const numAgents         = m.getParameter("numAgents",         NUM_AGENTS);

    const checkpointGovernance = m.contract(
      "CheckpointGovernance",
      [maxCapacityScaled, numAgents]
    );

    return { checkpointGovernance };
  }
);

export default CheckpointGovernanceModule;
