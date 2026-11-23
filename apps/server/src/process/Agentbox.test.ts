import * as NodeServices from "@effect/platform-node/NodeServices";
import { describe, expect, it } from "@effect/vitest";
import * as HostProcess from "@t3tools/shared/HostProcess";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";

import * as Agentbox from "./Agentbox.ts";

const input = {
  projects: ["/projects/cities", "/projects/fossil", "/projects/bicycle"],
  cpus: [4, 5, 6, 7, 8, 9, 10, 11],
  gpus: ["GPU-first", "GPU-second"],
  stateRoot: "/home/user/.agentbox/t3",
};

describe("Agentbox resource placement", () => {
  it.effect("rejects GPUs without a Slurm reservation or outside its visible allocation", () =>
    Effect.gen(function* () {
      for (const { environment, gpu } of [
        { environment: { SLURM_JOB_ID: "32469483", CUDA_VISIBLE_DEVICES: "0" }, gpu: "0" },
        {
          environment: { SLURM_JOB_ID: "42", SLURM_JOB_GPUS: "0", CUDA_VISIBLE_DEVICES: "0" },
          gpu: "1",
        },
      ]) {
        const error = yield* Effect.gen(function* () {
          const agentbox = yield* Agentbox.Agentbox;
          return yield* agentbox.run({
            mode: "run",
            layout: "/projects/layout.tsv",
            stateRoot: "/home/user/.agentbox/t3",
            cpuList: "0-1",
            gpuList: gpu,
            invocation: { command: "t3", args: [] },
          });
        }).pipe(
          Effect.provide(Agentbox.layer),
          Effect.provideService(HostProcess.Platform, "linux"),
          Effect.provideService(HostProcess.Environment, environment),
          Effect.provideService(
            FileSystem.FileSystem,
            FileSystem.makeNoop({
              readFileString: () => Effect.succeed("Cpus_allowed_list:\t0-1\n"),
            }),
          ),
          Effect.provide(NodeServices.layer),
          Effect.flip,
        );
        expect(error).toMatchObject({
          _tag: "AgentboxError",
          message:
            "Slurm GPU placement must stay within CUDA_VISIBLE_DEVICES; no login-node fallback.",
        });
      }
    }),
  );

  it("splits the allowed CPU set evenly and reports shared GPUs", () => {
    const plan = Agentbox.planAgentbox(input);
    expect(plan.map((session) => session.cpus.length)).toEqual([3, 3, 2]);
    expect(plan.flatMap((session) => session.cpus)).toEqual(input.cpus);
    expect(plan.map((session) => session.gpu)).toEqual(["GPU-first", "GPU-second", "GPU-first"]);
    expect(plan.map((session) => session.gpuSharedWith.length)).toEqual([1, 0, 1]);
  });

  it("keeps session identity, ports and state stable when the roster changes", () => {
    const first = Agentbox.planAgentbox(input);
    const second = Agentbox.planAgentbox({
      ...input,
      projects: [...input.projects, "/projects/new"],
    });
    for (const session of first) {
      const matching = second.find((candidate) => candidate.project === session.project);
      expect(matching).toMatchObject({
        id: session.id,
        port: session.port,
        stateDir: session.stateDir,
      });
    }
    expect(Agentbox.planAgentbox({ ...input, projects: [...input.projects].reverse() })).toEqual(
      first,
    );
    expect(
      Agentbox.planAgentbox({ ...input, projects: [...input.projects, input.projects[0]!] }),
    ).toEqual(first);
  });

  it("shares Friends' single GPU explicitly without claiming exclusive allocations", () => {
    const plan = Agentbox.planAgentbox({ ...input, gpus: ["GPU-3090"] });
    expect(plan.every((session) => session.gpu === "GPU-3090")).toBe(true);
    expect(plan.every((session) => session.gpuSharedWith.length === 2)).toBe(true);
  });

  it("rejects CPU-only and undersized allocations instead of using a login-node fallback", () => {
    expect(() => Agentbox.planAgentbox({ ...input, gpus: [] })).toThrow("GPU");
    expect(() => Agentbox.planAgentbox({ ...input, cpus: [0, 1] })).toThrow("CPU");
    expect(() => Agentbox.planAgentbox({ ...input, cpus: [0, 0, 0] })).toThrow("CPU");
    expect(() => Agentbox.planAgentbox({ ...input, projects: ["relative/path"] })).toThrow(
      "absolute",
    );
  });
});
