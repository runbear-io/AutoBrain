import { describe, expect, test } from "bun:test";
import { TOTAL_STAGE_WEIGHT } from "../data/diagnosis";
import { TRIALS } from "../data/optimization";
import { canRunDiagnosis, currentBestTrial, diagnosisPercent, initialState, optimizationPercent, reduce } from "./appState";

describe("prototype state journey", () => {
  test("diagnosis advances deterministically and caps at 100 percent", () => {
    const started = reduce(initialState(), { type: "diagnosis/start" });
    const finished = reduce(started, { type: "diagnosis/tick", amount: TOTAL_STAGE_WEIGHT + 20 });
    expect(finished.diagnosisComplete).toBe(true);
    expect(diagnosisPercent(finished)).toBe(100);
  });
  test("source reconnect makes the quick start launchable", () => {
    const state = initialState();
    const reconnected = reduce(state, { type: "source/reconnect", id: "notion" });
    expect(canRunDiagnosis(reconnected)).toBe(true);
    expect(reconnected.sources.find(source => source.id === "notion")?.stage).toBe("IMPORTED");
  });
  test("optimization completes all trials and selects trial nine", () => {
    let state = reduce(initialState(), { type: "optimization/start" });
    state = reduce(state, { type: "optimization/finish" });
    expect(state.completedTrials).toBe(TRIALS.length);
    expect(optimizationPercent(state)).toBe(100);
    expect(currentBestTrial(state)?.index).toBe(9);
  });
});
