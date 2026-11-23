import * as HostProcess from "@t3tools/shared/HostProcess";
import * as Console from "effect/Console";
import * as Effect from "effect/Effect";
import * as Option from "effect/Option";
import * as Schema from "effect/Schema";
import { Argument, Command, Flag } from "effect/cli";

import * as Agentbox from "../process/Agentbox.ts";

const encodeJson = Schema.encodeEffect(Schema.fromJsonString(Schema.Unknown));

export const agentboxCommand = Command.make("agentbox", {
  mode: Argument.Literals("operation", ["plan", "run", "serve"]),
  layout: Flag.String("layout").pipe(Flag.optional),
  stateRoot: Flag.String("state-root").pipe(Flag.optional),
  cpuList: Flag.String("cpu-list").pipe(Flag.optional),
  gpuList: Flag.String("gpu-list").pipe(Flag.optional),
  session: Flag.String("session").pipe(Flag.optional),
}).pipe(
  Command.withDescription(
    "Plan or run resource-placed project environments inside an existing GPU allocation.",
  ),
  Command.withHandler((flags) =>
    Effect.gen(function* () {
      const home = yield* HostProcess.HomeDirectory;
      const executable = yield* HostProcess.ExecutablePath;
      const args = (yield* HostProcess.IsExecutable) ? [] : [(yield* HostProcess.Arguments)[1]!];
      const agentbox = yield* Agentbox.Agentbox;
      const manifest = yield* agentbox.run({
        mode: flags.mode,
        layout: Option.getOrElse(flags.layout, () => `${home}/.config/gtab/layout.tsv`),
        stateRoot: Option.getOrElse(flags.stateRoot, () => `${home}/.agentbox/t3`),
        cpuList: Option.getOrUndefined(flags.cpuList),
        gpuList: Option.getOrUndefined(flags.gpuList),
        session: Option.getOrUndefined(flags.session),
        invocation: { command: executable, args },
      });
      yield* Console.log(yield* encodeJson(manifest));
    }).pipe(Effect.provide(Agentbox.layer)),
  ),
);
