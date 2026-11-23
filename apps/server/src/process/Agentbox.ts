import * as HostProcess from "@t3tools/shared/HostProcess";
import * as Context from "effect/Context";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Hash from "effect/Hash";
import * as Layer from "effect/Layer";
import * as Path from "effect/Path";
import type * as PlatformError from "effect/PlatformError";
import * as Schema from "effect/Schema";
import * as Stream from "effect/Stream";
import { ChildProcess, ChildProcessSpawner } from "effect/process";

const encodeJson = Schema.encodeEffect(Schema.fromJsonString(Schema.Unknown));

class AgentboxError extends Schema.TaggedError<AgentboxError>()("AgentboxError", {
  message: Schema.String,
}) {}

export function planAgentbox(input: {
  projects: ReadonlyArray<string>;
  cpus: ReadonlyArray<number>;
  gpus: ReadonlyArray<string>;
  stateRoot: string;
}) {
  const projects = [...new Set(input.projects)].sort();
  const cpus = [...new Set(input.cpus)];
  const gpus = [...new Set(input.gpus)];
  if (projects.some((project) => !project.startsWith("/")) || !input.stateRoot.startsWith("/")) {
    throw new AgentboxError({
      message: "Project folders and the state root must be absolute Linux paths.",
    });
  }
  if (cpus.some((cpu) => !Number.isSafeInteger(cpu) || cpu < 0) || cpus.length < projects.length) {
    throw new AgentboxError({
      message: "The allocation needs at least one allowed CPU per project session.",
    });
  }
  if (gpus.length === 0 || gpus.some((gpu) => !/^(?:\d+|GPU-[\w-]+|MIG-[\w/-]+)$/.test(gpu))) {
    throw new AgentboxError({
      message: "The allocation must expose at least one valid GPU identifier.",
    });
  }
  const ports = new Set<number>();
  let offset = 0;
  const plan = projects.map((project, index) => {
    const hash = Hash.string(project) >>> 0;
    const id = hash.toString(16).padStart(8, "0");
    const port = 20000 + (hash % 30000);
    if (ports.has(port))
      throw new AgentboxError({
        message: `Session port collision at ${port}; no sessions started.`,
      });
    ports.add(port);
    const count =
      Math.floor(cpus.length / projects.length) + (index < cpus.length % projects.length ? 1 : 0);
    const assigned = cpus.slice(offset, offset + count);
    offset += count;
    const stateDir = `${input.stateRoot.replace(/\/$/, "")}/${id}`;
    return {
      id,
      project,
      cpus: assigned,
      gpu: gpus[index % gpus.length]!,
      port,
      stateDir,
      serverArgs: [
        "--no-browser",
        "--host",
        "127.0.0.1",
        "--port",
        String(port),
        "--base-dir",
        stateDir,
        "--auto-bootstrap-project-from-cwd",
        project,
      ],
    };
  });
  return plan.map((session) => ({
    ...session,
    gpuSharedWith: plan
      .filter((other) => other.id !== session.id && other.gpu === session.gpu)
      .map((other) => other.id),
  }));
}

type AgentboxManifest = {
  readonly host: string;
  readonly jobId: string | null;
  readonly gpuSharing: string;
  readonly sessions: ReturnType<typeof planAgentbox>;
};

type AgentboxInput = {
  readonly mode: "plan" | "run" | "serve";
  readonly layout: string;
  readonly stateRoot: string;
  readonly cpuList?: string | undefined;
  readonly gpuList?: string | undefined;
  readonly session?: string | undefined;
  readonly invocation: { readonly command: string; readonly args: ReadonlyArray<string> };
};

export class Agentbox extends Context.Service<
  Agentbox,
  {
    readonly run: (
      input: AgentboxInput,
    ) => Effect.Effect<
      AgentboxManifest,
      AgentboxError | PlatformError.PlatformError | Schema.SchemaError
    >;
  }
>()("t3/process/Agentbox") {}

const make = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const environment = yield* HostProcess.Environment;
  const platform = yield* HostProcess.Platform;
  const hostname = yield* HostProcess.Hostname;
  const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;

  const run = Effect.fn("Agentbox.run")(function* (input: AgentboxInput) {
    const allowedCpuList =
      platform === "linux"
        ? /^Cpus_allowed_list:\s*(.+)$/m.exec(yield* fs.readFileString("/proc/self/status"))?.[1]
        : undefined;
    const [requestedCpus = [], allowedCpus = []] = [
      input.cpuList ?? allowedCpuList,
      allowedCpuList,
    ].map((list) =>
      (list ?? "")
        .split(",")
        .filter(Boolean)
        .flatMap((range) => {
          if (!/^\d+(?:-\d+)?$/.test(range))
            throw new AgentboxError({ message: `Invalid CPU range: ${range}` });
          const [start, end = start] = range.split("-").map(Number);
          if (start === undefined || end === undefined || end < start || end - start > 65535)
            throw new AgentboxError({ message: `Invalid CPU range: ${range}` });
          return Array.from({ length: end - start + 1 }, (_, index) => start + index);
        }),
    );
    if (
      input.mode !== "plan" &&
      (platform !== "linux" || requestedCpus.some((cpu) => !allowedCpus.includes(cpu)))
    ) {
      return yield* new AgentboxError({
        message: "Run requires Linux CPUs inside the current process's allowed CPU set.",
      });
    }
    const visibleGpus = environment.CUDA_VISIBLE_DEVICES?.split(",").filter(Boolean);
    const gpus = (input.gpuList ?? environment.CUDA_VISIBLE_DEVICES ?? "")
      .split(",")
      .filter(Boolean);
    if (
      environment.SLURM_JOB_ID &&
      (!environment.SLURM_JOB_GPUS ||
        !visibleGpus?.length ||
        gpus.some((gpu) => !visibleGpus.includes(gpu)))
    ) {
      return yield* new AgentboxError({
        message:
          "Slurm GPU placement must stay within CUDA_VISIBLE_DEVICES; no login-node fallback.",
      });
    }
    const folders = (yield* fs.readFileString(input.layout))
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
    const projects = yield* Effect.forEach(folders, (folder) => fs.realPath(folder));
    const plan = yield* Effect.try({
      try: () =>
        planAgentbox({
          projects,
          cpus: requestedCpus,
          gpus,
          stateRoot: path.resolve(input.stateRoot),
        }),
      catch: (cause) => new AgentboxError({ message: String(cause) }),
    });
    const manifest = {
      host: hostname,
      jobId: environment.SLURM_JOB_ID ?? null,
      gpuSharing: "shared access, not exclusive GPU memory or compute",
      sessions: plan,
    };
    if (input.mode === "plan") return manifest;
    const sessions =
      input.mode === "serve" ? plan : plan.filter((candidate) => candidate.id === input.session);
    if (sessions.length === 0)
      return yield* new AgentboxError({
        message:
          "Select a session from the plan with --session ID, or use serve for the complete roster.",
      });
    const gpuProbe = yield* spawner.spawn(
      ChildProcess.make(
        "nvidia-smi",
        ["--id", gpus.join(","), "--query-gpu=uuid", "--format=csv,noheader"],
        { stdout: "ignore", stderr: "inherit" },
      ),
    );
    if ((yield* gpuProbe.exitCode) !== 0)
      return yield* new AgentboxError({
        message: "Requested GPUs are unavailable on this machine; no sessions started.",
      });
    const stateRoot = path.resolve(input.stateRoot);
    yield* fs.makeDirectory(stateRoot, { recursive: true, mode: 0o700 });
    const children = yield* Effect.forEach(sessions, (session) =>
      Effect.gen(function* () {
        yield* fs.makeDirectory(session.stateDir, { recursive: true, mode: 0o700 });
        const log = path.join(session.stateDir, "server.log");
        yield* fs.writeFileString(log, "", { flag: "a", mode: 0o600 });
        yield* fs.chmod(log, 0o600);
        const child = yield* spawner.spawn(
          ChildProcess.make(
            "taskset",
            [
              "--cpu-list",
              session.cpus.join(","),
              input.invocation.command,
              ...input.invocation.args,
              ...session.serverArgs,
            ],
            {
              cwd: session.project,
              stdin: "ignore",
              stdout: "pipe",
              stderr: "pipe",
              env: {
                ...environment,
                CUDA_VISIBLE_DEVICES: session.gpu,
                OMP_NUM_THREADS: String(session.cpus.length),
                OPENBLAS_NUM_THREADS: String(session.cpus.length),
                MKL_NUM_THREADS: String(session.cpus.length),
                T3CODE_AGENTBOX_SESSION: session.id,
                T3CODE_ENVIRONMENT_LABEL: `${hostname}:${path.basename(session.project)}`,
              },
              extendEnv: false,
              detached: true,
              forceKillAfter: "5 seconds",
            },
          ),
        );
        yield* Stream.merge(child.stdout, child.stderr).pipe(
          Stream.run(fs.sink(log, { flag: "a" })),
          Effect.forkScoped,
        );
        return { session, child };
      }),
    );
    yield* fs.writeFileString(
      path.join(stateRoot, "manifest.json.new"),
      yield* encodeJson({ ...manifest, sessions }),
      { mode: 0o600 },
    );
    yield* fs.rename(
      path.join(stateRoot, "manifest.json.new"),
      path.join(stateRoot, "manifest.json"),
    );
    yield* Effect.logInfo(
      `Spawned ${children.length} project environments; resource manifest: ${path.join(stateRoot, "manifest.json")}`,
    );
    return yield* Effect.raceAll(
      children.map(({ session, child }) =>
        child.exitCode.pipe(
          Effect.flatMap((code) =>
            Effect.fail(
              new AgentboxError({
                message: `T3 session ${session.id} exited with ${code}; stopping this fleet's children.`,
              }),
            ),
          ),
        ),
      ),
    );
  }, Effect.scoped);

  return Agentbox.of({ run });
});

export const layer = Layer.effect(Agentbox, make);
