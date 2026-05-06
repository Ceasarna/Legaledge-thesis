// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract SmartChargingGovernance {

    address public owner;

    uint public peakHourStart = 17;
    uint public peakHourEnd   = 20;

    event EpisodeRecorded(
        uint indexed agentId,
        int totalEnergy,
        bool compliant
    );

    event PeakHoursUpdated(uint newStart, uint newEnd);

    struct EpisodeResult {
        int totalEnergy;
        bool compliant;
        uint timestamp;
    }

    mapping(uint => EpisodeResult[]) public episodeHistory;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    function setPeakHours(uint start, uint end) external onlyOwner {
        require(start < 24 && end < 24, "Hour out of range");
        require(start <= end, "Start must be <= end");
        peakHourStart = start;
        peakHourEnd   = end;
        emit PeakHoursUpdated(start, end);
    }

    function recordEpisode(
        uint agentId,
        int[] calldata powerKwScaled,
        uint[] calldata timestepHours
    ) external {
        uint n = powerKwScaled.length;
        require(n > 0, "Empty episode");
        require(timestepHours.length == n, "Array length mismatch");

        int totalEnergy;
        bool compliant = true;

        for (uint i = 0; i < n; i++) {
            require(timestepHours[i] < 24, "Invalid hour");
            totalEnergy += powerKwScaled[i];

            bool violation = (powerKwScaled[i] > 0)
                && (timestepHours[i] >= peakHourStart)
                && (timestepHours[i] <= peakHourEnd);

            if (violation) {
                compliant = false;
            }
        }

        episodeHistory[agentId].push(EpisodeResult({
            totalEnergy: totalEnergy,
            compliant:   compliant,
            timestamp:   block.timestamp
        }));

        emit EpisodeRecorded(agentId, totalEnergy, compliant);
    }

    function episodeCount(uint agentId) external view returns (uint) {
        return episodeHistory[agentId].length;
    }

    function getEpisodeResult(uint agentId, uint index)
        external view
        returns (int totalEnergy, bool compliant, uint timestamp)
    {
        require(index < episodeHistory[agentId].length, "Index out of bounds");
        EpisodeResult storage r = episodeHistory[agentId][index];
        return (r.totalEnergy, r.compliant, r.timestamp);
    }

    function lastCompliant(uint agentId) external view returns (bool) {
        uint len = episodeHistory[agentId].length;
        require(len > 0, "No episodes recorded");
        return episodeHistory[agentId][len - 1].compliant;
    }
}
