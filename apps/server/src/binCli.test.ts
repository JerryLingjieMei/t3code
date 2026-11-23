// @effect-diagnostics nodeBuiltinImport:off -- Exercises the CLI's real stdout and stderr through a subprocess and an HTTP fixture.
import * as NodeHttp from "node:http";
import * as NodeURL from "node:url";

import * as NodeServices from "@effect/platform-node/NodeServices";
import * as NodeCrypto from "@effect/platform-node/NodeCrypto";
import * as NetService from "@t3tools/shared/Net";
import { assert, it } from "@effect/vitest";
import * as NodeSqliteClient from "@t3tools/shared/nodeSqliteClient";
import * as Effect from "effect/Effect";
import * as FileSystem from "effect/FileSystem";
import * as Stream from "effect/Stream";
import * as Layer from "effect/Layer";
import * as Option from "effect/Option";
import * as Schema from "effect/Schema";
import { Command } from "effect/cli";
import { makeCli } from "./binCli.ts";
import { ChildProcess, ChildProcessSpawner } from "effect/process";
import * as SqlClient from "effect/sql/SqlClient";

import { migrationManifest, runMigrations } from "./persistence/Migrations.ts";
import {
  makePersistedServerRuntimeState,
  persistServerRuntimeState,
} from "./serverRuntimeState.ts";

const decodeAgentboxPlan = Schema.decodeEffect(
  Schema.fromJsonString(
    Schema.Struct({
      sessions: Schema.Array(
        Schema.Struct({
          serverArgs: Schema.Array(Schema.String),
          stateDir: Schema.String,
          project: Schema.String,
        }),
      ),
    }),
  ),
);

it.live("boots an Agentbox plan through the real CLI's project-bootstrap route", () =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3-agentbox-cli-" });
    const project = `${root}/project`;
    const layout = `${root}/layout.tsv`;
    yield* fs.makeDirectory(project);
    yield* fs.writeFileString(layout, `${project}\n`);
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
    const child = yield* spawner.spawn(
      ChildProcess.make(
        process.execPath,
        [
          NodeURL.fileURLToPath(new URL("./bin.ts", import.meta.url)),
          "agentbox",
          "plan",
          "--layout",
          layout,
          "--cpu-list",
          "0-1",
          "--gpu-list",
          "GPU-fixture",
          "--state-root",
          `${root}/t3-home`,
        ],
        { env: { SLURM_JOB_ID: "" } },
      ),
    );
    const { stdout, stderr, exitCode } = yield* Effect.all(
      {
        stdout: child.stdout.pipe(Stream.decodeText(), Stream.mkString),
        stderr: child.stderr.pipe(Stream.decodeText(), Stream.mkString),
        exitCode: child.exitCode,
      },
      { concurrency: "unbounded" },
    );
    assert.strictEqual(Number(exitCode), 0, stderr);
    const session = (yield* decodeAgentboxPlan(stdout)).sessions[0];
    assert(session !== undefined);
    let handled = false;
    const cli = makeCli({ cloudEnabled: false }).pipe(
      Command.withHandler((flags) =>
        Effect.sync(() => {
          handled = true;
          assert(
            "autoBootstrapProjectFromCwd" in flags &&
              "noBrowser" in flags &&
              "baseDir" in flags &&
              "cwd" in flags,
          );
          assert.deepStrictEqual(flags.autoBootstrapProjectFromCwd, Option.some(true));
          assert.deepStrictEqual(flags.noBrowser, Option.some(true));
          assert.deepStrictEqual(flags.baseDir, Option.some(session.stateDir));
          assert.deepStrictEqual(flags.cwd, Option.some(session.project));
        }),
      ),
    );
    yield* Command.runWith(cli, { version: "0.0.0" })(session.serverArgs);
    assert.strictEqual(handled, true);
  }).pipe(
    Effect.scoped,
    Effect.provide(Layer.mergeAll(NodeServices.layer, NodeCrypto.layer, NetService.layer)),
  ),
);

it.live("keeps migration warnings out of JSON thread output", () =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem;
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "t3-cli-json-" });
    const stateDir = `${root}/userdata`;
    yield* fs.makeDirectory(stateDir);
    yield* Effect.gen(function* () {
      const sql = yield* SqlClient.SqlClient;
      yield* runMigrations();
      const futureId = Math.max(...migrationManifest.map(([id]) => id)) + 1;
      yield* sql`
        INSERT INTO effect_sql_migrations (migration_id, name)
        VALUES (${futureId}, 'FutureMigration')
      `;
    }).pipe(Effect.provide(NodeSqliteClient.layer({ filename: `${stateDir}/statev2.sqlite` })));

    const server = yield* Effect.acquireRelease(
      Effect.promise(async () => {
        const server = NodeHttp.createServer(async (request, response) => {
          let body = "";
          for await (const chunk of request) body += String(chunk);
          if (request.method === "DELETE") {
            response.writeHead(204).end();
            return;
          }
          const message = JSON.parse(body) as {
            id?: string;
            method: string;
            params?: { name: string };
          };
          if (message.id === undefined) {
            response.writeHead(202).end();
            return;
          }
          const result =
            message.method === "initialize"
              ? { protocolVersion: "2025-06-18" }
              : {
                  structuredContent:
                    message.params?.name === "t3_project_list"
                      ? {
                          projects: [{ id: "project-1", title: "Test", workspaceRoot: root }],
                          nextCursor: null,
                        }
                      : {
                          projectId: "project-1",
                          currentThreadId: null,
                          threads: [],
                          nextCursor: null,
                          total: 0,
                        },
                };
          response.writeHead(200, {
            "content-type": "application/json",
            "mcp-session-id": "cli-json-test",
          });
          response.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
        });
        await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
        return server;
      }),
      (server) =>
        Effect.promise(() => new Promise<void>((resolve) => server.close(() => resolve()))),
    );
    const address = server.address();
    assert(address !== null && typeof address !== "string");
    yield* persistServerRuntimeState({
      path: `${stateDir}/server-runtime.json`,
      state: yield* makePersistedServerRuntimeState({
        config: { host: "127.0.0.1", devUrl: undefined },
        port: address.port,
      }),
    });

    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;
    const child = yield* spawner.spawn(
      ChildProcess.make(
        process.execPath,
        [
          NodeURL.fileURLToPath(new URL("./bin.ts", import.meta.url)),
          "thread",
          "list",
          "--project",
          "project-1",
          "--json",
          "--base-dir",
          root,
        ],
        { env: { T3CODE_HOME: root } },
      ),
    );
    const { stdout, stderr, exitCode } = yield* Effect.all(
      {
        stdout: child.stdout.pipe(Stream.decodeText(), Stream.mkString),
        stderr: child.stderr.pipe(Stream.decodeText(), Stream.mkString),
        exitCode: child.exitCode,
      },
      { concurrency: "unbounded" },
    );
    assert.strictEqual(Number(exitCode), 0, stderr);
    assert.deepStrictEqual(JSON.parse(stdout), []);
  }).pipe(Effect.scoped, Effect.provide(NodeServices.layer)),
);
