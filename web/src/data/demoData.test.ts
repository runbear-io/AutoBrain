import { describe, expect, test } from "bun:test";
import { DIAGNOSIS_RUN, winner } from "./diagnosis";
import { BEST_TRIAL, TRIALS, bestFeasibleTrial, paretoFrontier } from "./optimization";

describe("synthetic evidence fixtures", () => {
  test("GBrain wins the diagnosis by the declared policy", () => {
    expect(DIAGNOSIS_RUN.verdict).toBe("gbrain");
    expect(winner(DIAGNOSIS_RUN).id).toBe("gbrain");
  });
  test("twelve trials produce the declared feasible winner", () => {
    expect(TRIALS).toHaveLength(12);
    expect(bestFeasibleTrial(TRIALS)?.index).toBe(BEST_TRIAL.index);
  });
  test("Pareto frontier excludes pruned and violating trials", () => {
    const frontier = paretoFrontier(TRIALS);
    expect(frontier.length).toBeGreaterThan(1);
    expect(frontier.every(trial => trial.state === "complete" && !trial.violatesConstraint)).toBe(true);
  });
});
