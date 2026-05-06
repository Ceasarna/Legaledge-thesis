"""
Standalone Ray throughput benchmark.

This script intentionally bypasses Flower. It launches one Ray actor per
logical agent, starts all actors from the same wall-clock barrier, records
per-agent timing/process/memory metrics, and writes a CSV comparable to the
Flower throughput CSV.

Usage:
    python -m src.analysis.throughput_ray_standalone --dry-run
"""

import argparse
import csv
import math
import os
import platform
import statistics
import time
from typing import Dict, Iterable, List, Optional

import ray

from src.federated.config import SMART_CONTRACT_ADDRESS

try:
    import psutil
except ImportError:
    psutil = None


CSV_COLUMNS = [
    "round",
    "agent_id",
    "pid",
    "round_wall_start",
    "round_wall_end",
    "first_tx_wall_s",
    "last_tx_wall_s",
    "n_blockchain_tx",
    "rss_mb",
    "mean_tx_latency_ms",
    "p50_tx_latency_ms",
    "p95_tx_latency_ms",
    "max_tx_latency_ms",
    "mean_gas_used",
    "total_gas_used",
]


def submit_blockchain_tx(
    agent_id: int,
    round_id: int,
    tx_idx: int,
    dry_run: bool,
    tx_sleep_s: float,
    tx_retries: int,
    tx_retry_sleep_s: float,
    fail_on_tx_error: bool,
    bridge=None,
    power_log: Optional[Dict[int, float]] = None,
) -> Optional[Dict[str, float]]:
    """Hook for the blockchain transaction under benchmark."""
    if dry_run:
        if tx_sleep_s > 0:
            time.sleep(tx_sleep_s)
        return {"latency": tx_sleep_s * 1000.0, "gasUsed": 0.0}

    if bridge is None:
        raise RuntimeError("A blockchain bridge is required when --dry-run is not set")

    # SmartChargingGovernance benchmark hook:
    # this is the same contract path used by the single-scenario client callback,
    # but without importing Flower or running RL training.
    result = None
    for attempt in range(tx_retries + 1):
        result = bridge.record_episode(agent_id, power_log or make_power_log())
        if result is not None:
            break
        if attempt < tx_retries and tx_retry_sleep_s > 0:
            time.sleep(tx_retry_sleep_s)

    if result is None:
        message = (
            "SmartChargingGovernance recordEpisode returned no result "
            f"(agent_id={agent_id}, round_id={round_id}, tx_idx={tx_idx}). "
            "Check that Hardhat is running, the contract address matches the "
            "current deployment, and the transaction did not revert."
        )
        if fail_on_tx_error:
            raise RuntimeError(message)
        return None

    if tx_sleep_s > 0:
        time.sleep(tx_sleep_s)
    return result


def make_power_log(episode_hours: int = 24, power_kw: float = 7.0) -> Dict[int, float]:
    """Build a deterministic smart-charging episode payload for recordEpisode."""
    return {
        hour: (0.0 if 17 <= hour <= 20 else power_kw)
        for hour in range(episode_hours)
    }


def _rss_mb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


@ray.remote
class AgentActor:
    def __init__(self, agent_id: int, config: Dict):
        self.agent_id = agent_id
        self.config = config
        self.pid = os.getpid()
        self.bridge = None
        self.power_log = make_power_log(
            int(config["episode_hours"]),
            float(config["power_kw"]),
        )
        if not bool(config["dry_run"]):
            self._get_bridge()

    def ping(self) -> Dict[str, int]:
        return {"agent_id": self.agent_id, "pid": self.pid}

    def _get_bridge(self):
        if self.bridge is None:
            from src.governance.contract_bridge import ContractBridge

            self.bridge = ContractBridge(
                self.config["contract_address"],
                agent_id=self.agent_id,
                rpc_url=self.config["rpc_url"],
            )
        return self.bridge

    def run_round(
        self,
        round_id: int,
        start_at: float,
        tx_per_agent: int,
        tx_sleep_s: float,
        dry_run: bool,
    ) -> Dict[str, float]:
        sleep_s = start_at - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

        round_wall_start = time.time()
        first_tx_wall_s = 0.0
        last_tx_wall_s = 0.0
        n_blockchain_tx = 0
        n_failed_tx = 0
        latencies_ms = []
        gas_used_values = []

        for tx_idx in range(tx_per_agent):
            tx_start = time.time()
            if tx_idx == 0:
                first_tx_wall_s = tx_start

            result = submit_blockchain_tx(
                self.agent_id,
                round_id,
                tx_idx,
                dry_run=dry_run,
                tx_sleep_s=tx_sleep_s,
                tx_retries=int(self.config["tx_retries"]),
                tx_retry_sleep_s=float(self.config["tx_retry_sleep_s"]),
                fail_on_tx_error=bool(self.config["fail_on_tx_error"]),
                bridge=None if dry_run else self._get_bridge(),
                power_log=self.power_log,
            )
            if result is not None:
                n_blockchain_tx += 1
                latencies_ms.append(float(result.get("latency", 0.0)))
                gas_used_values.append(float(result.get("gasUsed", 0.0)))
            else:
                n_failed_tx += 1
            last_tx_wall_s = time.time()

        round_wall_end = time.time()
        if tx_per_agent == 0:
            first_tx_wall_s = round_wall_start
            last_tx_wall_s = round_wall_end

        return {
            "round": round_id,
            "agent_id": self.agent_id,
            "pid": self.pid,
            "round_wall_start": round_wall_start,
            "round_wall_end": round_wall_end,
            "first_tx_wall_s": first_tx_wall_s,
            "last_tx_wall_s": last_tx_wall_s,
            "n_blockchain_tx": n_blockchain_tx,
            "n_failed_tx": n_failed_tx,
            "rss_mb": _rss_mb(),
            "mean_tx_latency_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50_tx_latency_ms": percentile(latencies_ms, 0.50),
            "p95_tx_latency_ms": percentile(latencies_ms, 0.95),
            "max_tx_latency_ms": max(latencies_ms) if latencies_ms else 0.0,
            "mean_gas_used": statistics.mean(gas_used_values) if gas_used_values else 0.0,
            "total_gas_used": sum(gas_used_values),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone Ray actor throughput benchmark."
    )
    parser.add_argument("--n-agents", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--tx-per-agent", type=int, default=420)
    parser.add_argument("--cpus-per-agent", type=float, default=0.25)
    parser.add_argument("--ray-num-cpus", type=int, default=24)
    parser.add_argument("--ray-num-gpus", type=int, default=0)
    parser.add_argument("--object-store-memory", type=int, default=None)
    parser.add_argument("--start-delay-s", type=float, default=5.0)
    parser.add_argument("--tx-sleep-s", type=float, default=0.0)
    parser.add_argument("--output", default="ray_throughput.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pending", type=int, default=None)
    parser.add_argument("--contract-address", default=SMART_CONTRACT_ADDRESS)
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--power-kw", type=float, default=7.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-min-agents", type=int, default=2)
    parser.add_argument("--sweep-max-agents", type=int, default=512)
    parser.add_argument("--tx-retries", type=int, default=2)
    parser.add_argument("--tx-retry-sleep-s", type=float, default=0.1)
    parser.add_argument("--fail-on-tx-error", action="store_true")
    parser.add_argument("--stop-sweep-on-error", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_agents < 1:
        raise ValueError("--n-agents must be >= 1")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.tx_per_agent < 0:
        raise ValueError("--tx-per-agent must be >= 0")
    if args.cpus_per_agent <= 0:
        raise ValueError("--cpus-per-agent must be > 0")
    if args.ray_num_cpus <= 0:
        raise ValueError("--ray-num-cpus must be > 0")
    if args.ray_num_gpus < 0:
        raise ValueError("--ray-num-gpus must be >= 0")
    if args.start_delay_s < 0:
        raise ValueError("--start-delay-s must be >= 0")
    if args.tx_sleep_s < 0:
        raise ValueError("--tx-sleep-s must be >= 0")
    if args.object_store_memory is not None and args.object_store_memory <= 0:
        raise ValueError("--object-store-memory must be > 0 when provided")
    if args.max_pending is not None and args.max_pending < 1:
        raise ValueError("--max-pending must be >= 1 when provided")
    if not args.dry_run and not args.contract_address:
        raise ValueError("--contract-address is required when --dry-run is not set")
    if args.episode_hours < 1 or args.episode_hours > 24:
        raise ValueError("--episode-hours must be between 1 and 24")
    if args.power_kw < 0:
        raise ValueError("--power-kw must be >= 0")
    if args.sweep_min_agents < 1:
        raise ValueError("--sweep-min-agents must be >= 1")
    if args.sweep_max_agents < args.sweep_min_agents:
        raise ValueError("--sweep-max-agents must be >= --sweep-min-agents")
    if args.tx_retries < 0:
        raise ValueError("--tx-retries must be >= 0")
    if args.tx_retry_sleep_s < 0:
        raise ValueError("--tx-retry-sleep-s must be >= 0")


def init_ray(args: argparse.Namespace) -> None:
    ray_kwargs = {
        "num_cpus": args.ray_num_cpus,
        "num_gpus": args.ray_num_gpus,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    if args.object_store_memory is not None:
        ray_kwargs["object_store_memory"] = args.object_store_memory

    ray.init(**ray_kwargs)


def get_rpc_account_count(rpc_url: str) -> int:
    from web3 import Web3

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to RPC endpoint: {rpc_url}")
    return len(w3.eth.accounts)


def validate_blockchain_capacity(args: argparse.Namespace) -> None:
    if args.dry_run:
        return

    account_count = get_rpc_account_count(args.rpc_url)
    print(f"RPC unlocked accounts: {account_count}")
    if args.n_agents > account_count and args.max_pending is None:
        raise ValueError(
            f"--n-agents={args.n_agents} exceeds the {account_count} unlocked "
            "RPC accounts. ContractBridge chooses accounts by agent_id modulo "
            "account_count, so concurrent actors will reuse sender accounts and "
            "can race on nonces. Start Hardhat with at least one account per "
            "agent, reduce --n-agents, or use --max-pending no larger than the "
            "account count for a throttled run."
        )


def validate_ray_scheduling_capacity(args: argparse.Namespace) -> None:
    theoretical_max = math.floor(args.ray_num_cpus / args.cpus_per_agent)
    print(f"theoretical max concurrency: {theoretical_max}")
    if theoretical_max < args.n_agents:
        raise ValueError(
            f"Ray cannot schedule {args.n_agents} actors concurrently with "
            f"--ray-num-cpus={args.ray_num_cpus} and "
            f"--cpus-per-agent={args.cpus_per_agent}. "
            "Increase --ray-num-cpus or lower --cpus-per-agent. For 512 agents "
            "with 24 Ray CPUs, use --cpus-per-agent 0.046875 or lower."
        )


def create_actors(args: argparse.Namespace) -> List:
    config = {
        "cpus_per_agent": args.cpus_per_agent,
        "ray_num_cpus": args.ray_num_cpus,
        "ray_num_gpus": args.ray_num_gpus,
        "contract_address": args.contract_address,
        "rpc_url": args.rpc_url,
        "episode_hours": args.episode_hours,
        "power_kw": args.power_kw,
        "dry_run": args.dry_run,
        "tx_retries": args.tx_retries,
        "tx_retry_sleep_s": args.tx_retry_sleep_s,
        "fail_on_tx_error": args.fail_on_tx_error,
    }
    return [
        AgentActor.options(num_cpus=args.cpus_per_agent, num_gpus=0).remote(
            agent_id, config
        )
        for agent_id in range(args.n_agents)
    ]


def collect_round_results(
    actors: List,
    round_id: int,
    start_at: float,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    if args.max_pending is None:
        refs = [
            actor.run_round.remote(
                round_id,
                start_at,
                args.tx_per_agent,
                args.tx_sleep_s,
                args.dry_run,
            )
            for actor in actors
        ]
        return ray.get(refs)

    pending = []
    results = []
    actor_iter = iter(actors)

    def submit_next() -> bool:
        try:
            actor = next(actor_iter)
        except StopIteration:
            return False
        pending.append(
            actor.run_round.remote(
                round_id,
                start_at,
                args.tx_per_agent,
                args.tx_sleep_s,
                args.dry_run,
            )
        )
        return True

    while len(pending) < args.max_pending and submit_next():
        pass

    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        results.extend(ray.get(ready))
        while len(pending) < args.max_pending and submit_next():
            pass

    return results


def peak_concurrency(rows: Iterable[Dict[str, float]]) -> int:
    events = []
    for row in rows:
        events.append((row["round_wall_start"], 0, 1))
        events.append((row["round_wall_end"], 1, -1))

    current = 0
    peak = 0
    for _, _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["round"], item["agent_id"])):
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def resolve_output_path(output: str, n_agents: int, force_agent_suffix: bool = False) -> str:
    if output != "ray_throughput.csv" and not force_agent_suffix:
        return output
    stem, ext = os.path.splitext(output)
    return f"{stem}_{n_agents}_agents{ext}"


def sweep_agent_counts(min_agents: int, max_agents: int) -> List[int]:
    counts = []
    n_agents = min_agents
    while n_agents <= max_agents:
        counts.append(n_agents)
        n_agents *= 2
    return counts


def print_summary(args: argparse.Namespace, rows: List[Dict[str, float]]) -> None:
    durations = [row["round_wall_end"] - row["round_wall_start"] for row in rows]
    first_tx_times = [row["first_tx_wall_s"] for row in rows]
    last_tx_times = [row["last_tx_wall_s"] for row in rows]
    unique_pids = {row["pid"] for row in rows}
    peak = peak_concurrency(rows)
    theoretical_max = math.floor(args.ray_num_cpus / args.cpus_per_agent)

    benchmark_wall_time = max(row["round_wall_end"] for row in rows) - min(
        row["round_wall_start"] for row in rows
    )
    attempted_tx = args.n_agents * args.rounds * args.tx_per_agent
    successful_tx = sum(int(row["n_blockchain_tx"]) for row in rows)
    failed_tx = sum(int(row.get("n_failed_tx", 0)) for row in rows)
    throughput = successful_tx / benchmark_wall_time if benchmark_wall_time > 0 else 0.0
    latency_means = [
        float(row["mean_tx_latency_ms"])
        for row in rows
        if int(row["n_blockchain_tx"]) > 0
    ]
    latency_p95s = [
        float(row["p95_tx_latency_ms"])
        for row in rows
        if int(row["n_blockchain_tx"]) > 0
    ]
    total_gas = sum(float(row["total_gas_used"]) for row in rows)
    mean_gas_values = [
        float(row["mean_gas_used"])
        for row in rows
        if int(row["n_blockchain_tx"]) > 0
    ]

    print("\nRay standalone throughput benchmark")
    print("=" * 43)
    print(f"total wall-clock time: {benchmark_wall_time:.6f} s")
    print(f"peak concurrency: {peak}")
    print(f"unique PID count: {len(unique_pids)}")
    print(f"attempted transactions: {attempted_tx}")
    print(f"successful transactions: {successful_tx}")
    print(f"failed transactions: {failed_tx}")
    print(f"tx throughput: {throughput:.6f} tx/s")
    print(
        "tx latency mean/p95/max-agent-p95: "
        f"{statistics.mean(latency_means) if latency_means else 0.0:.3f} / "
        f"{percentile(latency_p95s, 0.95):.3f} / "
        f"{max(latency_p95s) if latency_p95s else 0.0:.3f} ms"
    )
    print(f"mean gas/tx: {statistics.mean(mean_gas_values) if mean_gas_values else 0.0:.2f}")
    print(f"total gas used: {total_gas:.0f}")
    print(f"theoretical max concurrency: {theoretical_max}")
    print(
        "per-agent duration min/mean/p50/p95/max: "
        f"{min(durations):.6f} / "
        f"{statistics.mean(durations):.6f} / "
        f"{percentile(durations, 0.50):.6f} / "
        f"{percentile(durations, 0.95):.6f} / "
        f"{max(durations):.6f} s"
    )
    print(f"first tx span: {max(first_tx_times) - min(first_tx_times):.6f} s")
    print(f"last tx span: {max(last_tx_times) - min(last_tx_times):.6f} s")

    if peak < args.n_agents:
        print(
            "WARNING: Ray did not run all agents concurrently. "
            "Check ray-num-cpus / cpus-per-agent / OS limits."
        )


def run_benchmark_for_agent_count(args: argparse.Namespace, n_agents: int) -> None:
    original_n_agents = args.n_agents
    args.n_agents = n_agents
    output_path = resolve_output_path(
        args.output,
        args.n_agents,
        force_agent_suffix=args.sweep,
    )

    actors = []
    try:
        validate_ray_scheduling_capacity(args)
        validate_blockchain_capacity(args)
        actors = create_actors(args)
        ping_results = ray.get([actor.ping.remote() for actor in actors])
        print(f"created logical agents: {len(ping_results)}")

        rows = []
        for round_id in range(args.rounds):
            start_at = time.time() + args.start_delay_s
            rows.extend(collect_round_results(actors, round_id, start_at, args))

        print_summary(args, rows)
        write_csv(output_path, rows)
        print(f"wrote CSV: {output_path}")
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        args.n_agents = original_n_agents


def main() -> None:
    args = parse_args()
    validate_args(args)

    if platform.system() == "Windows":
        print(
            "Ray on native Windows can behave differently under high process counts. "
            "For thesis-grade throughput numbers, prefer WSL2/Linux."
        )

    try:
        init_ray(args)
        print(f"ray.cluster_resources(): {ray.cluster_resources()}")
        print(f"ray.available_resources(): {ray.available_resources()}")

        if args.sweep:
            counts = sweep_agent_counts(args.sweep_min_agents, args.sweep_max_agents)
            print(f"sweep agent counts: {counts}")
            for n_agents in counts:
                print("\n" + "#" * 72)
                print(f"Starting sweep run for n_agents={n_agents}")
                print("#" * 72)
                try:
                    run_benchmark_for_agent_count(args, n_agents)
                except Exception as exc:
                    print(f"ERROR: sweep run failed for n_agents={n_agents}: {exc}")
                    if args.stop_sweep_on_error:
                        raise
                    print("Continuing to next sweep size.")
        else:
            run_benchmark_for_agent_count(args, args.n_agents)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
