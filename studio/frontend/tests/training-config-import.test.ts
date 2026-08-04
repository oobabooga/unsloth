// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Save/Load config buttons in the training section round-trip a YAML file through
// serializeConfigToYaml -> parseYamlConfig -> store.applyConfigPatch. These drive that
// last call, the real store action the file input reaches, rather than the mapper alone:
// a patch that looks right in isolation can still wreck the rest of the store.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { BackendModelConfig } from "../src/features/training/api/models-api.ts";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();

const yaml = await import("js-yaml");
const { useTrainingConfigStore } = await import(
  "../src/features/training/stores/training-config-store.ts"
);
const { parseYamlConfig, serializeConfigToYaml } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);

// A shipped config with a hand-tuned recipe that disagrees with the global defaults on
// most fields, and the reason the boolean gradient_checkpointing case below is not
// hypothetical: 7 of the 78 shipped configs write that key as a YAML boolean.
const TUNED_MODEL_CONFIG = new URL(
  "../../backend/assets/configs/model_defaults/llama/unsloth_Llama-3.2-1B-Instruct.yaml",
  import.meta.url,
);

/** Seed the store the way loadAndApplyModelDefaults does once a model is picked. */
function seedTunedModelDefaults(): void {
  const config = yaml.load(
    readFileSync(TUNED_MODEL_CONFIG, "utf8"),
  ) as BackendModelConfig;
  useTrainingConfigStore.setState(mapBackendModelConfigToTrainingPatch(config));
}

/** Every value in the store, so a test can name exactly which ones an import moved. */
function snapshot(): Record<string, unknown> {
  const state = useTrainingConfigStore.getState() as unknown as Record<
    string,
    unknown
  >;
  return Object.fromEntries(
    Object.entries(state).filter(([, value]) => typeof value !== "function"),
  );
}

function keysChangedBy(action: () => void): string[] {
  const before = snapshot();
  action();
  const after = snapshot();
  return Object.keys(after).filter(
    (key) => !Object.is(after[key], before[key]),
  );
}

function importConfig(text: string): void {
  useTrainingConfigStore.getState().applyConfigPatch(parseYamlConfig(text));
}

test("a partial import patches only the keys the file names", () => {
  seedTunedModelDefaults();

  const changed = keysChangedBy(() =>
    importConfig("training:\n  max_seq_length: 4096\n"),
  );

  assert.deepEqual(
    changed,
    ["contextLength"],
    "a file naming one key must not reset the selected model's other tuned values",
  );
  assert.equal(useTrainingConfigStore.getState().contextLength, 4096);
});

test("a tuned model recipe survives an unrelated import", () => {
  seedTunedModelDefaults();
  const tuned = snapshot();

  // Not the global defaults: those are lr 2e-4, batch 4, adamw_8bit, linear,
  // train_on_completions off, lora_alpha 32. Importing must not slide them there.
  assert.equal(tuned.learningRate, 2e-5);
  assert.equal(tuned.batchSize, 1);
  assert.equal(tuned.optimizerType, "adamw_torch");
  assert.equal(tuned.lrSchedulerType, "cosine");
  assert.equal(tuned.trainOnCompletions, true);
  assert.equal(tuned.loraAlpha, 16);

  importConfig("lora:\n  lora_r: 64\n");

  const after = useTrainingConfigStore.getState();
  assert.equal(after.loraRank, 64, "the imported key applies");
  assert.equal(after.learningRate, 2e-5);
  assert.equal(after.batchSize, 1);
  assert.equal(after.optimizerType, "adamw_torch");
  assert.equal(after.lrSchedulerType, "cosine");
  assert.equal(after.trainOnCompletions, true);
  assert.equal(after.loraAlpha, 16);
});

test("gradient_checkpointing is read from a YAML boolean as well as a string", () => {
  seedTunedModelDefaults();
  assert.equal(
    useTrainingConfigStore.getState().gradientCheckpointing,
    "true",
    "the shipped config says gradient_checkpointing: true, unquoted",
  );

  importConfig("training:\n  gradient_checkpointing: false\n");
  assert.equal(useTrainingConfigStore.getState().gradientCheckpointing, "none");

  importConfig("training:\n  gradient_checkpointing: unsloth\n");
  assert.equal(
    useTrainingConfigStore.getState().gradientCheckpointing,
    "unsloth",
  );
});

test("a blank number is treated as absent, not as zero", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ epochs: 5, warmupSteps: 7 });

  importConfig('training:\n  num_epochs: ""\n  warmup_steps: "   "\n');

  const after = useTrainingConfigStore.getState();
  assert.equal(after.epochs, 5);
  assert.equal(after.warmupSteps, 7);

  importConfig("training:\n  num_epochs: 0\n");
  assert.equal(
    useTrainingConfigStore.getState().epochs,
    0,
    "a real 0 still applies; only the blank is ignored",
  );
});

test("saving and reloading a config keeps the logging settings", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({
    enableWandb: true,
    wandbProject: "my-project",
    enableTensorboard: true,
    tensorboardDir: "my-runs",
    logFrequency: 25,
  });

  const saved = serializeConfigToYaml(useTrainingConfigStore.getState(), false);

  useTrainingConfigStore.setState({
    enableWandb: false,
    wandbProject: "",
    enableTensorboard: false,
    tensorboardDir: "",
    logFrequency: 1,
  });
  importConfig(saved);

  const after = useTrainingConfigStore.getState();
  assert.equal(after.enableWandb, true);
  assert.equal(after.wandbProject, "my-project");
  assert.equal(after.enableTensorboard, true);
  assert.equal(after.tensorboardDir, "my-runs");
  assert.equal(after.logFrequency, 25);
});

test("saving and reloading a config keeps the embedding learning rate", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ embeddingLearningRate: 3e-5 });

  const saved = serializeConfigToYaml(useTrainingConfigStore.getState(), false);

  useTrainingConfigStore.setState({ embeddingLearningRate: null });
  importConfig(saved);
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, 3e-5);

  // null means "let the backend derive it", so it has to round-trip too rather
  // than leave the previous rate standing.
  useTrainingConfigStore.setState({ embeddingLearningRate: null });
  const clearedSave = serializeConfigToYaml(
    useTrainingConfigStore.getState(),
    false,
  );
  useTrainingConfigStore.setState({ embeddingLearningRate: 9e-5 });
  importConfig(clearedSave);
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, null);
});

test("a file with no embedding learning rate leaves the current one alone", () => {
  seedTunedModelDefaults();
  useTrainingConfigStore.setState({ embeddingLearningRate: 4e-5 });

  importConfig("training:\n  max_seq_length: 4096\n");
  assert.equal(useTrainingConfigStore.getState().embeddingLearningRate, 4e-5);
});
