import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

export default buildModule("SmartChargingGovernanceModule", (m) => {
  const governance = m.contract("SmartChargingGovernance");
  return { governance };
});
