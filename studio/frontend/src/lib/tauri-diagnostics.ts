// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { redactDiagnosticsText } from "@/lib/diagnostics-redaction";

export {
  createDiagnosticsLineRedactor,
  latestRedactedDiagnosticsLine,
  redactDiagnosticsLineStream,
  redactDiagnosticsText,
  type DiagnosticsLineStreamState,
} from "@/lib/diagnostics-redaction";

export interface FrontendSupportSnapshot {
  status?: string | null;
  error?: string | null;
  currentStepIndex?: number | null;
  progressDetail?: string | null;
  elevationPackages?: string[];
  lastUiLogLines?: string[];
  flow?: string | null;
  updatePhase?: string | null;
  updateProgress?: number | null;
}

export interface CopySupportDiagnosticsResult {
  ok: boolean;
  report: string;
  error?: string;
  source: "tauri" | "frontend-fallback";
}

const FALLBACK_LOG_LINE_LIMIT = 200;

function safeLines(lines: string[] | undefined): string[] {
  return (lines ?? [])
    .filter((line): line is string => typeof line === "string")
    .slice(-FALLBACK_LOG_LINE_LIMIT);
}

function buildFrontendFallbackReport(
  snapshot: FrontendSupportSnapshot,
  invokeError: unknown,
): string {
  const lines = safeLines(snapshot.lastUiLogLines);
  const report = [
    "DEGRADED FRONTEND-ONLY DIAGNOSTICS",
    "DEGRADED_FRONTEND_ONLY=true",
    `createdAt=${new Date().toISOString()}`,
    "reason=collect_support_diagnostics invoke failed",
    `invokeError=${String(invokeError)}`,
    `status=${snapshot.status ?? "unavailable"}`,
    `flow=${snapshot.flow ?? "unknown"}`,
    `error=${snapshot.error ?? ""}`,
    `currentStepIndex=${snapshot.currentStepIndex ?? ""}`,
    `progressDetail=${snapshot.progressDetail ?? ""}`,
    `elevationPackages=${(snapshot.elevationPackages ?? []).join(",")}`,
    `updatePhase=${snapshot.updatePhase ?? ""}`,
    `updateProgress=${snapshot.updateProgress ?? ""}`,
    "",
    "lastUiLogLines:",
    ...lines,
    "",
    "redaction=frontend_minimal_js_redaction_applied",
    "collectionWarnings=Rust diagnostics command was unavailable; this fallback contains only bounded UI state/log lines.",
  ].join("\n");

  return redactDiagnosticsText(report);
}

async function collectDiagnosticsReport(
  snapshot: FrontendSupportSnapshot,
): Promise<{ report: string; source: CopySupportDiagnosticsResult["source"] }> {
  if (!isTauri) {
    return {
      report: buildFrontendFallbackReport(snapshot, "not running in Tauri"),
      source: "frontend-fallback",
    };
  }

  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const report = await invoke<string>("collect_support_diagnostics", {
      snapshot,
    });
    return { report, source: "tauri" };
  } catch (error) {
    console.warn("collect_support_diagnostics failed; using frontend-only diagnostics", error);
    return {
      report: buildFrontendFallbackReport(snapshot, error),
      source: "frontend-fallback",
    };
  }
}

async function copyWithTauriClipboard(text: string): Promise<boolean> {
  if (!isTauri) return false;
  try {
    const { writeText } = await import("@tauri-apps/plugin-clipboard-manager");
    await writeText(text);
    return true;
  } catch (error) {
    console.warn("Tauri clipboard-manager writeText failed", error);
    return false;
  }
}

export async function copySupportDiagnostics(
  snapshot: FrontendSupportSnapshot,
): Promise<CopySupportDiagnosticsResult> {
  const { report, source } = await collectDiagnosticsReport(snapshot);

  if (await copyWithTauriClipboard(report)) {
    return { ok: true, report, source };
  }

  if (await copyToClipboard(report)) {
    return { ok: true, report, source };
  }

  return {
    ok: false,
    report,
    source,
    error: "Unable to write diagnostics to the clipboard. Select and copy the text manually.",
  };
}
