import json
import threading
import time
from pathlib import Path
from web3 import Web3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCALE = 1000

_ARTIFACT_CANDIDATES = (
    PROJECT_ROOT / "artifacts" / "contracts",
    PROJECT_ROOT / "src" / "governance" / "config" / "artifacts" / "contracts",
)

def _artifact_path(*parts: str) -> Path:
    for base in _ARTIFACT_CANDIDATES:
        candidate = base.joinpath(*parts)
        if candidate.exists():
            return candidate
    return _ARTIFACT_CANDIDATES[0].joinpath(*parts)

# ── Process-local transaction log ────────────────────────────────────────
# Every transact + wait_for_transaction_receipt records (start_wall, end_wall, kind).
# The client reads this at the end of each round to compute throughput windows
# without relying on chain timestamps (which Hardhat instant-mining inflates by 1s/block).
_TX_LOG: list = []
_TX_LOG_LOCK = threading.Lock()

def _record_tx(start_wall: float, end_wall: float, kind: str) -> None:
    with _TX_LOG_LOCK:
        _TX_LOG.append((start_wall, end_wall, kind))

def read_and_clear_tx_log() -> list:
    with _TX_LOG_LOCK:
        out = list(_TX_LOG)
        _TX_LOG.clear()
        return out

# Individual Governance Contract Web3 Bridge
class ContractBridge:
    def __init__(self, contract_address: str, agent_id: int = 0,
                 rpc_url: str = "http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Cannot connect to Hardhat node"

        with open(_artifact_path("SmartChargingGovernance.sol", "SmartChargingGovernance.json")) as f:
            abi = json.load(f)["abi"]

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )

        accounts = self.w3.eth.accounts
        self.account = accounts[agent_id % len(accounts)]

    def record_episode(self, agent_id: int, power_log: dict) -> dict:
        sorted_hours   = sorted(power_log.keys())
        power_scaled   = [int(power_log[h] * 100) for h in sorted_hours]
        timestep_hours = sorted_hours

        t0 = time.perf_counter()
        wall_start = time.time()

        try:
            transaction = self.contract.functions.recordEpisode(
                agent_id, power_scaled, timestep_hours
            ).transact({"from": self.account, "gas": 1000000})

            receipt = self.w3.eth.wait_for_transaction_receipt(transaction)
            t1 = time.perf_counter()
            _record_tx(wall_start, time.time(), "record_episode")

            gas_used = int(receipt["gasUsed"])
            total_ms = (t1 - t0) * 1000

            logs = self.contract.events.EpisodeRecorded().process_receipt(receipt)
            args = logs[0]["args"]
            return {
                "agentId":     int(args["agentId"]),
                "totalEnergy": int(args["totalEnergy"]) / 100,
                "compliant":   bool(args["compliant"]),
                "gasUsed":     gas_used,
                "latency":     total_ms,
            }

        except Exception as e:
            print(f"Error in record_episode: {e}")
            return None

    def get_peak_hours(self) -> tuple[int, int]:
        """Read the current peak-hour window from the contract.

        This is a read-only `call` — no gas cost, no transaction, no state change.
        Returns (peak_start, peak_end) as 0-23 hour integers (inclusive range).
        """
        start = self.contract.functions.peakHourStart().call()
        end = self.contract.functions.peakHourEnd().call()
        return int(start), int(end)

    def submit_batch(self, agent_results: list) -> dict:
        raise NotImplementedError("submit_batch not available in test contract")


# Collective Governance Contract Web3 Bridge
class CheckpointGovernanceBridge:
    """
    Bridge for the CheckpointGovernance contract.
    Agents call submit_checkpoint() every N timesteps during an episode.
    The contract returns the collective total across all agents.
    """

    ABI_PATH = _artifact_path("CheckpointGovernance.sol", "CheckpointGovernance.json")

    def __init__(self, contract_address: str, agent_id: int = 0,
                 rpc_url: str = "http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        assert self.w3.is_connected(), "Cannot connect to Hardhat node"

        with open(self.ABI_PATH) as f:
            abi = json.load(f)["abi"]

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )

        accounts = self.w3.eth.accounts
        self.account = accounts[agent_id % len(accounts)]
        self.max_capacity_scaled = self.contract.functions.maxCapacityScaled().call()
        self.num_agents = self.contract.functions.numExpectedAgents().call()

    def submit_checkpoint(self, round_id, episode_id, checkpoint_id, agent_id, cumulative_energy_kwh):
        # 1. Check if episode is already closed
        if self.contract.functions.isFinalized(round_id, episode_id).call():
            total_norm = self.get_checkpoint_total(round_id, episode_id, checkpoint_id)
            grid_max_kwh = self.max_capacity_scaled / 1000.0
            total_kwh = total_norm * grid_max_kwh
            ratio = (cumulative_energy_kwh / total_kwh) if total_kwh > 0 else 0
            return total_norm, ratio

        # 2. Submit transaction to blockchain
        energy_scaled = int(cumulative_energy_kwh * 1000)
        wall_start = time.time()
        tx_hash = self.contract.functions.submitCheckpoint(
            round_id, episode_id, checkpoint_id, agent_id, energy_scaled
        ).transact({"from": self.account, "gas": 300_000})
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        _record_tx(wall_start, time.time(), "submit_checkpoint")

        # 3. Synchronize: Wait for all agents to finish their submissions
        self._wait_for_sync(round_id, episode_id, checkpoint_id)

        # 4. Fetch final consensus total
        final_total_norm = self.get_checkpoint_total(round_id, episode_id, checkpoint_id)
        
        # 5. Calculate ratio based on final collective state
        grid_max_kwh = self.max_capacity_scaled / 1000.0
        final_total_kwh = final_total_norm * grid_max_kwh
        
        true_ratio = (cumulative_energy_kwh / final_total_kwh) if final_total_kwh > 0 else 0
        
        return final_total_norm, true_ratio

    def _wait_for_sync(self, r_id, e_id, c_id):
        count = self.contract.functions.getCheckpointCount(r_id, e_id, c_id).call()
        while count < self.num_agents:
            import time
            time.sleep(0.01)
            count = self.contract.functions.getCheckpointCount(r_id, e_id, c_id).call()

    def get_checkpoint_total(
        self, round_id: int, episode_id: int, checkpoint_id: int
    ) -> float:
        # Read the collective total
        total = self.contract.functions.getCheckpointTotal(
            round_id, episode_id, checkpoint_id
        ).call()
        return total / max(self.max_capacity_scaled, 1)

    def finalize_episode(
            self, round_id: int, episode_id: int, final_checkpoint_id: int
        ) -> dict:
            # Finalize an episode
            # 1. READ CHECK: See if another agent already finalized this episode
            is_already_finalized = self.contract.functions.isFinalized(round_id, episode_id).call()

            if not is_already_finalized:
                try:
                    # 2. WRITE: Only the first agent to reach here sends the transaction
                    wall_start = time.time()
                    tx = self.contract.functions.finalizeEpisode(
                        round_id, episode_id, final_checkpoint_id
                    ).transact({"from": self.account, "gas": 200_000})
                    self.w3.eth.wait_for_transaction_receipt(tx)
                    _record_tx(wall_start, time.time(), "finalize_episode")
                except Exception as e:
                    print(f"[Bridge] Finalize race condition handled: {e}")

            # 3. RETURN RESULT: Fetch the current state
            finalized, penalized, total_energy = (
                self.contract.functions.getEpisodeResult(round_id, episode_id).call()
            )
            
            return {
                "finalized": finalized,
                "penalized": penalized,
                "total_energy": total_energy / SCALE,
                "total_energy_norm": total_energy / max(self.max_capacity_scaled, 1),
            }