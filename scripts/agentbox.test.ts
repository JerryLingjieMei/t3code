import { describe, expect, it } from "vite-plus/test";

import { planAgentbox } from "./agentbox.ts";

const input = {
  projects: ["/projects/cities", "/projects/fossil", "/projects/bicycle"],
  cpus: [4, 5, 6, 7, 8, 9, 10, 11],
  gpus: ["GPU-first", "GPU-second"],
  stateRoot: "/home/user/.agentbox/t3",
};

describe("Agentbox resource placement", () => {
  it("splits the allowed CPU set evenly and reports shared GPUs", () => {
    const plan = planAgentbox(input);
    expect(plan.map((session) => session.cpus.length)).toEqual([3, 3, 2]);
    expect(plan.flatMap((session) => session.cpus)).toEqual(input.cpus);
    expect(plan.map((session) => session.gpu)).toEqual(["GPU-first", "GPU-second", "GPU-first"]);
    expect(plan.map((session) => session.gpuSharedWith.length)).toEqual([1, 0, 1]);
  });

  it("keeps session identity, ports and state stable when the roster changes", () => {
    const first = planAgentbox(input);
    const second = planAgentbox({ ...input, projects: [...input.projects, "/projects/new"] });
    for (const session of first) {
      const matching = second.find((candidate) => candidate.project === session.project);
      expect(matching).toMatchObject({
        id: session.id,
        port: session.port,
        stateDir: session.stateDir,
      });
    }
    expect(planAgentbox({ ...input, projects: [...input.projects].reverse() })).toEqual(first);
    expect(planAgentbox({ ...input, projects: [...input.projects, input.projects[0]!] })).toEqual(
      first,
    );
  });

  it("shares Friends' single GPU explicitly without claiming exclusive allocations", () => {
    const plan = planAgentbox({ ...input, gpus: ["GPU-3090"] });
    expect(plan.every((session) => session.gpu === "GPU-3090")).toBe(true);
    expect(plan.every((session) => session.gpuSharedWith.length === 2)).toBe(true);
  });

  it("rejects CPU-only and undersized allocations instead of using a login-node fallback", () => {
    expect(() => planAgentbox({ ...input, gpus: [] })).toThrow("GPU");
    expect(() => planAgentbox({ ...input, cpus: [0, 1] })).toThrow("CPU");
    expect(() => planAgentbox({ ...input, cpus: [0, 0, 0] })).toThrow("CPU");
    expect(() => planAgentbox({ ...input, projects: ["relative/path"] })).toThrow("absolute");
  });
});
