// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/**
 * @title CheckpointGovernance
 * @notice Mid-episode coordination contract for federated EV charging.
 *
 * Agents submit their cumulative energy at periodic checkpoints (e.g. every
 * 4 timesteps within a 24-step episode).  The contract aggregates all agents'
 * submissions for each checkpoint and exposes the collective total so that
 * agents can observe real-time grid utilisation and self-limit.
 *
 * At the end of the episode (final checkpoint or explicit finalize call) the
 * contract evaluates whether the collective total exceeded the capacity and
 * records the compliance result on-chain for auditability.
 */
contract CheckpointGovernance {

    uint public immutable maxCapacityScaled;   // e.g. 50_000 = 50.0 kWh × 1000
    uint public immutable numExpectedAgents;
    uint public constant  SCALE = 1000;

    // roundId → episodeId → checkpointId → agentId → cumulative energy (scaled)
    mapping(uint => mapping(uint => mapping(uint => mapping(uint => uint))))
        public agentCumulativeEnergy;

    // roundId → episodeId → checkpointId → agentId → has submitted?
    mapping(uint => mapping(uint => mapping(uint => mapping(uint => bool))))
        public hasSubmittedCheckpoint;

    // roundId → episodeId → checkpointId → number of agents that submitted
    mapping(uint => mapping(uint => mapping(uint => uint)))
        public checkpointSubmitted;

    // roundId → episodeId → checkpointId → total energy across all agents
    mapping(uint => mapping(uint => mapping(uint => uint)))
        public checkpointTotalEnergy;

    struct EpisodeResult {
        uint totalEnergy;     
        bool finalized;
        bool penalized;       
    }

    // roundId → episodeId → result
    mapping(uint => mapping(uint => EpisodeResult)) public episodes;

    event CheckpointSubmitted(
        uint indexed roundId,
        uint indexed episodeId,
        uint checkpointId,
        uint indexed agentId,
        uint cumulativeEnergy,
        uint collectiveTotal,
        uint contributionRatio
    );

    event EpisodeFinalized(
        uint indexed roundId,
        uint indexed episodeId,
        uint totalEnergy,
        bool penalized
    );
    
    constructor(uint _maxCapacityScaled, uint _numAgents) {
        require(_numAgents > 0 && _maxCapacityScaled > 0, "invalid params");
        maxCapacityScaled = _maxCapacityScaled;
        numExpectedAgents = _numAgents;
    }


    /**
     * @notice Submit cumulative energy at a checkpoint.
     * @param roundId      FL round number
     * @param episodeId    Episode index within the round
     * @param checkpointId Checkpoint index within the episode (0, 1, 2, ...)
     * @param agentId      Agent identifier (0-indexed)
     * @param cumulativeEnergy  Total energy charged so far this episode (scaled)
     * @return collectiveTotal  Sum of all agents' cumulative energy at this checkpoint
     */
    function submitCheckpoint(
        uint roundId,
        uint episodeId,
        uint checkpointId,
        uint agentId,
        uint cumulativeEnergy
    ) external returns (uint collectiveTotal, uint contributionRatio) {
        require(agentId < numExpectedAgents, "bad agentId");
        require(!episodes[roundId][episodeId].finalized, "episode already finalized");

        // If first time for this checkpoint, increment counter
        if (!hasSubmittedCheckpoint[roundId][episodeId][checkpointId][agentId]) {
            hasSubmittedCheckpoint[roundId][episodeId][checkpointId][agentId] = true;
            checkpointSubmitted[roundId][episodeId][checkpointId] += 1;
        }

        // Store cumulative energy for this agent at this checkpoint
        agentCumulativeEnergy[roundId][episodeId][checkpointId][agentId] = cumulativeEnergy;

        // Recompute total across all agents for this checkpoint
        uint total = 0;
        for (uint i = 0; i < numExpectedAgents; i++) {
            total += agentCumulativeEnergy[roundId][episodeId][checkpointId][i];
        }
        checkpointTotalEnergy[roundId][episodeId][checkpointId] = total;

        uint myRatio = 0;
        if (total > 0){
            myRatio = (cumulativeEnergy * SCALE) / total;
        }

        emit CheckpointSubmitted(
            roundId, episodeId, checkpointId, agentId, cumulativeEnergy, total, myRatio
        );

        return (total, myRatio);
    }

    /**
     * @notice Get the collective total at a specific checkpoint.
     *         Can be called by agents that didn't trigger the submission.
     */
    function getCheckpointTotal(
        uint roundId, uint episodeId, uint checkpointId
    ) external view returns (uint) {
        return checkpointTotalEnergy[roundId][episodeId][checkpointId];
    }

    /**
     * @notice How many agents have submitted at a given checkpoint.
     */
    function getCheckpointCount(
        uint roundId, uint episodeId, uint checkpointId
    ) external view returns (uint) {
        return checkpointSubmitted[roundId][episodeId][checkpointId];
    }

    // Episode finalization
    /**
     * @notice Finalize an episode using a specific checkpoint as the final total.
     * @param finalCheckpointId  The last checkpoint of the episode
     */
    function finalizeEpisode(
        uint roundId,
        uint episodeId,
        uint finalCheckpointId
    ) external {
        require(!episodes[roundId][episodeId].finalized, "already finalized");

        uint total = checkpointTotalEnergy[roundId][episodeId][finalCheckpointId];
        episodes[roundId][episodeId].totalEnergy = total;
        episodes[roundId][episodeId].finalized = true;
        episodes[roundId][episodeId].penalized = total > maxCapacityScaled;

        emit EpisodeFinalized(roundId, episodeId, total, total > maxCapacityScaled);
    }

    // Episode result reader
    function getEpisodeResult(uint roundId, uint episodeId)
        external view
        returns (bool finalized, bool penalized, uint totalEnergy)
    {
        EpisodeResult storage r = episodes[roundId][episodeId];
        return (r.finalized, r.penalized, r.totalEnergy);
    }

    function isFinalized(uint roundId, uint episodeId) external view returns (bool) {
        return episodes[roundId][episodeId].finalized;
    }
}
