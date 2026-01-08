#!/usr/bin/env node

import React from "react";
import { render } from "ink";
import { Command } from "commander";
import { App } from "./App.js";
import { listRuns, loadRunData } from "./lib/parser.js";

const program = new Command();

program
  .name("extty")
  .description("Terminal UI for ML experiment tracking")
  .version("0.1.0");

program
  .command("watch [run-name]", { isDefault: true })
  .description("Watch training runs in real-time")
  .action((runName?: string) => {
    render(<App initialRun={runName} />);
  });

program
  .command("list")
  .description("List all runs")
  .action(() => {
    const runs = listRuns();
    if (runs.length === 0) {
      console.log("No runs found in ~/.extty/runs/");
      return;
    }

    console.log("Runs:\n");
    for (const run of runs) {
      const data = loadRunData(run);
      const status = data.meta?.status ?? "unknown";
      const project = data.meta?.project ?? "?";
      const statusColor =
        status === "running" ? "\x1b[32m" : status === "completed" ? "\x1b[34m" : "\x1b[33m";
      console.log(`  ${statusColor}●\x1b[0m ${run} (${project}) - ${status}`);
    }
  });

program.parse();
