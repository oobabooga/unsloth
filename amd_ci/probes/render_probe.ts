// Render THIS checkout's real Resources tab against a fixture of device readings
// and report the text it produces. State agnostic on purpose: the same file runs
// against the PR base and the PR head, so the base arm is a real render of the old
// component rather than "the new hook could not be imported", which would make the
// differential vacuous.
//
// Copied into <checkout>/studio/frontend/tests/ before it runs, because it loads
// ./helpers/module-stubs.ts and ../src from there.
//
// Usage:
//   node --experimental-strip-types tests/render_probe.ts --cases cases.json --out out.json

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { argv } from "node:process";
import { fileURLToPath } from "node:url";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import * as gpuVram from "../src/hooks/gpu-vram.ts";
import { en } from "../src/i18n/locales/en.ts";

// URL, not a path string: the two are joined with `new URL` below, and
// url.pathname on Windows is "/C:/..." which existsSync cannot open.
const SRC = new URL("../src/", import.meta.url);

// messages.ts's own pattern and lookup rules, against the shipped English catalogue:
// the strings under test are the real ones, and a missing key returns the key so a
// renamed translation shows up as a key in the output rather than as silence.
const PLACEHOLDER_PATTERN = /\{([a-zA-Z0-9_]+)\}/g;
function t(key: string, values?: Record<string, unknown>): string {
  let cursor: unknown = en;
  for (const part of key.split(".")) {
    if (typeof cursor !== "object" || cursor === null) return key;
    cursor = (cursor as Record<string, unknown>)[part];
  }
  if (typeof cursor !== "string") return key;
  if (!values) return cursor;
  return cursor.replace(PLACEHOLDER_PATTERN, (match, name: string) =>
    Object.prototype.hasOwnProperty.call(values, name)
      ? values[name] === null || values[name] === undefined
        ? ""
        : String(values[name])
      : match,
  );
}

type Props = Record<string, unknown> & { children?: React.ReactNode };
/** A stand-in that keeps its children in the tree, so their text still renders. */
const passthrough = (tag: string) => (props: Props) =>
  React.createElement(tag, { "data-stub": props["aria-label"] ?? undefined }, props.children ?? null);
const nothing = () => null;
/** A zustand-shaped hook over a fixed state. */
const store = (state: Record<string, unknown>) => (selector?: (s: unknown) => unknown) =>
  selector ? selector(state) : state;

function renderResources(systemInfo: unknown): string {
  const stubs: Record<string, unknown> = {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/button": { Button: passthrough("button") },
    "@/components/ui/input": { Input: passthrough("span") },
    // The bar is the thing the PR suppresses, so it must be observable in the markup.
    "@/components/ui/progress": {
      Progress: (props: Props) =>
        React.createElement("div", {
          "data-progress": String(props.value ?? ""),
          "aria-label": String(props["aria-label"] ?? ""),
        }),
    },
    "@/components/ui/switch": { Switch: nothing },
    "@/features/hub": { formatBytes: (n: number) => `${n} B` },
    "@/features/model-picker": { FolderBrowser: nothing },
    "@/features/native-intents": {
      openModelsDir: async () => undefined,
      pickHuggingFaceCacheDir: async () => null,
    },
    "@/hooks/gpu-vram": gpuVram,
    "@/hooks/use-system": {
      aggregateGpuMemoryTotalGb: gpuVram.aggregateGpuMemoryTotalGb,
      useSystemInfo: () => systemInfo,
    },
    "@/lib/api-base": { isTauri: false },
    "@/lib/copy-to-clipboard": { copyToClipboard: async () => true },
    "@/lib/toast": { toast: { success: nothing, error: nothing } },
    "@/lib/utils": { cn: (...parts: unknown[]) => parts.filter(Boolean).join(" ") },
    "@/i18n": { useT: () => t },
    "../api/hugging-face-cache": {
      loadHuggingFaceCacheSettings: async () => null,
      updateHuggingFaceCacheSettings: async () => null,
    },
    "../components/llama-backend-section": { LlamaBackendSection: nothing },
    "../components/model-memory-section": { ModelMemorySection: nothing },
    "../components/settings-row": { SettingsRow: passthrough("div") },
    "../components/settings-section": {
      SettingsSection: (props: Props) =>
        React.createElement("section", null, String(props.title ?? ""), props.children ?? null),
    },
    "../stores/monitor-overlay-store": {
      useMonitorOverlayStore: store({ isOpen: false, setIsOpen: nothing }),
    },
    "../stores/settings-panel-prefs-store": {
      useSettingsPanelPrefsStore: store({
        resourcesLiveUpdates: false,
        setResourcesLiveUpdates: nothing,
      }),
    },
    "lucide-react": new Proxy({}, { get: () => nothing }),
  };
  // Present at the head, absent at the base. Loaded here rather than stubbed, so the
  // head arm exercises the PR's own hook and not a copy of it.
  const hookPath = new URL("hooks/gpu-memory-display.ts", SRC);
  if (existsSync(fileURLToPath(hookPath))) {
    stubs["@/hooks/gpu-memory-display"] = loadWithStubs(hookPath, {});
  }

  const mod = loadWithStubs<{ ResourcesTab: React.ComponentType }>(
    new URL("features/settings/tabs/resources-tab.tsx", SRC),
    stubs,
  );
  return renderToStaticMarkup(React.createElement(mod.ResourcesTab));
}

function renderMonitor(systemInfo: unknown): string {
  const motionProxy = new Proxy(
    {},
    {
      get: (_target, tag: string) => (props: Props) =>
        React.createElement("div", { "data-motion": tag }, props.children ?? null),
    },
  );
  const stubs: Record<string, unknown> = {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/components/ui/button": { Button: passthrough("button") },
    "@/components/ui/progress": {
      Progress: (props: Props) =>
        React.createElement("div", {
          "data-progress": String(props.value ?? ""),
          "aria-label": String(props["aria-label"] ?? ""),
        }),
    },
    "@/features/find-in-page/lib/find-attributes": { FIND_PORTAL_ATTRIBUTE: "data-find-portal" },
    "@/features/settings": {
      useMonitorOverlayStore: store({ isOpen: true, setIsOpen: nothing }),
      useMonitorFrameStore: store({
        width: 320,
        height: 420,
        setSize: nothing,
        position: { x: 0, y: 0 },
        setPosition: nothing,
        reset: nothing,
      }),
    },
    "@/hooks/gpu-vram": gpuVram,
    "@/hooks/use-system": {
      aggregateGpuMemoryTotalGb: gpuVram.aggregateGpuMemoryTotalGb,
      useSystemInfo: () => systemInfo,
    },
    "@/i18n": { useT: () => t },
    "@/lib/floating-panel-order": {
      useFloatingPanelOrderStore: store({ raise: nothing }),
      useFloatingPanelZIndex: () => 10,
    },
    "@/lib/utils": { cn: (...parts: unknown[]) => parts.filter(Boolean).join(" ") },
    "lucide-react": new Proxy({}, { get: () => nothing }),
    "motion/react": { AnimatePresence: passthrough("div"), motion: motionProxy },
  };
  const hookPath = new URL("hooks/gpu-memory-display.ts", SRC);
  if (existsSync(fileURLToPath(hookPath))) {
    stubs["@/hooks/gpu-memory-display"] = loadWithStubs(hookPath, {});
  }
  const mod = loadWithStubs<{ FloatingMonitor: React.ComponentType }>(
    new URL("components/floating-monitor.tsx", SRC),
    stubs,
  );
  return renderToStaticMarkup(React.createElement(mod.FloatingMonitor));
}

/** Markup to the visible sentences, so a class-name change is not mistaken for a text change. */
function visibleText(markup: string): string {
  return markup
    .replace(/<[^>]*>/g, "\u0001")
    .replace(/&#x27;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&")
    .split("\u0001")
    .map((s) => s.trim())
    .filter(Boolean)
    .join(" | ");
}

function progressBars(markup: string): string[] {
  return [...markup.matchAll(/data-progress="([^"]*)"\s*aria-label="([^"]*)"/g)].map(
    (m) => `${m[2]}=${m[1]}`,
  );
}

function main(): number {
  const casesPath = argv[argv.indexOf("--cases") + 1];
  const outPath = argv[argv.indexOf("--out") + 1];
  const cases = JSON.parse(readFileSync(casesPath, "utf8")) as {
    name: string;
    systemInfo: unknown;
  }[];

  const results: Record<string, unknown> = {
    hook_present: existsSync(fileURLToPath(new URL("hooks/gpu-memory-display.ts", SRC))),
    cases: {},
  };
  for (const c of cases) {
    const entry: Record<string, unknown> = {};
    for (const [surface, render] of [
      ["resources", renderResources],
      ["monitor", renderMonitor],
    ] as const) {
      try {
        const markup = render(c.systemInfo);
        entry[surface] = { text: visibleText(markup), bars: progressBars(markup) };
      } catch (error) {
        entry[surface] = { error: `${(error as Error).name}: ${(error as Error).message}` };
      }
    }
    (results.cases as Record<string, unknown>)[c.name] = entry;
  }
  writeFileSync(outPath, JSON.stringify(results, null, 2), "utf8");
  return 0;
}

process.exitCode = main();
