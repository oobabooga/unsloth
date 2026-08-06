// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  INITIAL_STARTUP_MESSAGE,
  installProgressMessage,
  SERVER_STARTUP_MESSAGE,
  SERVER_START_FALLBACK_MS,
  type StartupMessage,
} from "@/components/tauri/startup-messages";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { Spinner } from "@/components/ui/spinner";
import type { BackendStatus } from "@/hooks/use-tauri-backend";
import {
  latestRedactedDiagnosticsLine,
  redactDiagnosticsLineStream,
  redactDiagnosticsText,
  type CopySupportDiagnosticsResult,
  type DiagnosticsLineStreamState,
} from "@/lib/tauri-diagnostics";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";

interface StartupScreenProps {
  status: BackendStatus;
  logs: string[];
  error: string | null;
  currentStepIndex: number;
  progressDetail: string | null;
  startupMessage: StartupMessage;
  elevationPackages: string[];
  onInstall: () => void;
  onRetry: () => void;
  onRetryInstall: () => void;
  onApproveElevation: () => void;
  onStartServer: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}

function DiagnosticsCopyActions({
  onCopyDiagnostics,
  children,
}: {
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
  children: React.ReactNode;
}) {
  const [copying, setCopying] = useState(false);
  const [manualReport, setManualReport] = useState<string | null>(null);
  const [manualMessage, setManualMessage] = useState<string | null>(null);

  async function handleCopyDiagnostics() {
    setCopying(true);
    try {
      const result = await onCopyDiagnostics();
      if (result.ok) {
        setManualReport(null);
        setManualMessage(null);
      } else {
        setManualReport(result.report);
        setManualMessage(result.error ?? "Clipboard copy failed. Select and copy the diagnostics below.");
      }
    } catch (error) {
      setManualReport(null);
      setManualMessage(`Diagnostics copy failed: ${String(error)}`);
    } finally {
      setCopying(false);
    }
  }

  return (
    <div className="mt-4 flex w-full flex-col items-center gap-3">
      <div className="flex gap-3">
        <ActionButton
          variant="secondary"
          onClick={() => void handleCopyDiagnostics()}
        >
          {copying ? "Copying..." : "Copy Diagnostics"}
        </ActionButton>
        {children}
      </div>
      {manualMessage && (
        <p className="max-w-md text-center text-xs text-destructive">{manualMessage}</p>
      )}
      {manualReport && (
        <textarea
          readOnly
          value={manualReport}
          onFocus={(event) => event.currentTarget.select()}
          className="h-32 w-full max-w-md resize-none rounded-lg border border-border/50 bg-muted/30 p-2 font-mono text-ui-10 text-muted-foreground"
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------


const EASE_OUT_QUART: [number, number, number, number] = [0.165, 0.84, 0.44, 1];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Logo() {
  return (
    <div className="flex items-center justify-center gap-3">
      <img
        src="/sticker.png"
        alt=""
        aria-hidden="true"
        className="h-[60px] w-[60px] object-contain"
      />
      <span
        className="text-ui-50 font-semibold leading-none tracking-[-0.02em] text-foreground"
        style={{ fontFamily: '"Hellix", sans-serif' }}
      >
        unsloth
      </span>
    </div>
  );
}

function ActionButton({
  onClick,
  variant = "primary",
  children,
}: {
  onClick: () => void;
  variant?: "primary" | "secondary";
  children: React.ReactNode;
}) {
  const base = "rounded-lg px-5 py-2.5 text-sm font-medium cursor-pointer transition-colors";
  const styles =
    variant === "primary"
      ? `${base} bg-primary text-primary-foreground hover:bg-primary/80`
      : `${base} bg-muted text-foreground hover:bg-muted/80`;
  return (
    <button type="button" className={styles} onClick={onClick}>
      {children}
    </button>
  );
}

const STARTUP_LOG_PREVIEW_LIMIT = 1_000;

// Caps a single progress line only. Error text is never truncated: the startup
// timeout message carries the backend's last output, which is the whole point of it.
function boundedLogPreview(line: string) {
  if (line.length <= STARTUP_LOG_PREVIEW_LIMIT) return line;
  return `${line.slice(0, STARTUP_LOG_PREVIEW_LIMIT)}…`;
}

function useLatestVisibleLog(logs: string[]) {
  return useMemo(
    () => boundedLogPreview(latestRedactedDiagnosticsLine(logs)),
    [logs],
  );
}

function useRedactedLogDetails(
  logs: string[],
  progressDetail: string | null,
): string[] {
  const redactionState = useRef<DiagnosticsLineStreamState | undefined>(
    undefined,
  );
  const rawLines = useMemo(
    () => (progressDetail ? [...logs, progressDetail] : logs),
    [logs, progressDetail],
  );
  return useMemo(() => {
    const next = redactDiagnosticsLineStream(
      rawLines,
      redactionState.current,
    );
    redactionState.current = next;
    return next.displayedLines.map(boundedLogPreview);
  }, [rawLines]);
}

function visibleLines(text: string) {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .join("\n");
}

function usePeriodicValue<T>(value: T, intervalMs: number) {
  const latestRef = useRef(value);
  const [published, setPublished] = useState(value);

  useEffect(() => {
    latestRef.current = value;
  }, [value]);

  useEffect(() => {
    const id = setInterval(() => setPublished(latestRef.current), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return published;
}

// ---------------------------------------------------------------------------
// Per-status renderers
// ---------------------------------------------------------------------------

function CheckingContent() {
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm text-muted-foreground">Checking...</p>
      </div>
    </div>
  );
}

function NotInstalledContent({ onInstall }: { onInstall: () => void }) {
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 flex-col items-center justify-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-3">
        <p
          className="text-ui-13 font-semibold tracking-[-0.01em] text-muted-foreground"
          style={{ fontFamily: '"Hellix", sans-serif' }}
        >
          To install Unsloth, click Get Started.
        </p>
        <ShimmerButton
          onClick={onInstall}
          shimmerColor="#a7f3d0"
          background="oklch(0.696 0.17 162.48)"
          className="text-sm font-medium"
        >
          Get Started
        </ShimmerButton>
      </div>
    </div>
  );
}

function InstallingContent({
  logs,
  currentStepIndex,
  progressDetail,
}: {
  logs: string[];
  currentStepIndex: number;
  progressDetail: string | null;
}) {
  const message = installProgressMessage(currentStepIndex);
  const detailLines = useRedactedLogDetails(logs, progressDetail);

  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex w-full flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm font-bold text-foreground" aria-live="polite">
          {message.title}
        </p>
        <p className="text-sm text-muted-foreground">{message.subtitle}</p>
        {detailLines.length > 0 && (
          <details className="group mt-2 w-full max-w-sm text-left">
            <summary className="mx-auto w-fit cursor-pointer select-none rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <span className="group-open:hidden">Show installation details</span>
              <span className="hidden group-open:inline">Hide installation details</span>
            </summary>
            <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground">
              {detailLines.join("\n")}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

function RepairingContent({
  logs,
  progressDetail,
}: {
  logs: string[];
  progressDetail: string | null;
}) {
  // Repair logs are pip/uv output from the managed install and can carry index
  // URLs with credentials, tokens and home paths, exactly like startup logs.
  const detailLines = useRedactedLogDetails(logs, progressDetail);

  return (
    <div className="flex h-full w-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex w-full flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm font-bold text-foreground">Getting things ready...</p>
        <p className="text-sm text-muted-foreground">This won’t take long.</p>
        {detailLines.length > 0 && (
          <details className="group mt-2 w-full max-w-sm text-left">
            <summary className="mx-auto w-fit cursor-pointer select-none rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <span className="group-open:hidden">Show setup details</span>
              <span className="hidden group-open:inline">Hide setup details</span>
            </summary>
            <pre className="mt-2 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/50 bg-muted/30 p-3 font-mono text-ui-10 leading-relaxed text-muted-foreground">
              {detailLines.join("\n")}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}

function InstallErrorContent({
  error,
  onRetryInstall,
  onCopyDiagnostics,
}: {
  error: string | null;
  onRetryInstall: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  const displayedError = error ? redactDiagnosticsText(error) : null;

  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Setup ran into a problem</p>
        {displayedError && (
          <p className="max-w-xs whitespace-pre-wrap break-words text-center text-xs text-muted-foreground">
            {displayedError}
          </p>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetryInstall}>Try Again</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </>
  );
}

function RepairErrorContent({
  error,
  onRetry,
  onCopyDiagnostics,
}: {
  error: string | null;
  onRetry: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  const displayedError = error ? redactDiagnosticsText(error) : null;

  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-destructive">Update failed</p>
        {displayedError && (
          <p className="max-w-md whitespace-pre-wrap break-words text-center text-xs text-muted-foreground">
            {displayedError}
          </p>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </>
  );
}

function NeedsElevationContent({
  elevationPackages,
  onApproveElevation,
  onRetryInstall,
}: {
  elevationPackages: string[];
  onApproveElevation: () => void;
  onRetryInstall: () => void;
}) {
  const displayedPackages = useRedactedLogDetails(elevationPackages, null);

  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-foreground">Permission needed</p>
        <p className="text-xs text-muted-foreground">
          The following system packages need to be installed:
        </p>
        <div className="mt-2 w-full max-w-xs rounded-lg bg-muted p-3 font-mono text-xs">
          {displayedPackages.map((pkg, index) => (
            <div key={`${index}:${pkg}`}>{pkg}</div>
          ))}
        </div>
        <div className="mt-4 flex gap-3">
          <ActionButton variant="secondary" onClick={onRetryInstall}>Cancel</ActionButton>
          <ActionButton onClick={onApproveElevation}>Allow</ActionButton>
        </div>
      </div>
    </>
  );
}

function StartingContent({
  message,
  logs,
}: {
  message: StartupMessage;
  logs: string[];
}) {
  const [showFallback, setShowFallback] = useState(false);
  const latestLog = useLatestVisibleLog(logs);
  const announcedLog = usePeriodicValue(latestLog, 2_000);

  useEffect(() => {
    if (message !== SERVER_STARTUP_MESSAGE) {
      return;
    }

    const fallback = window.setTimeout(
      () => setShowFallback(true),
      SERVER_START_FALLBACK_MS,
    );
    return () => window.clearTimeout(fallback);
  }, [message]);

  const displayMessage = showFallback ? INITIAL_STARTUP_MESSAGE : message;
  return (
    <div className="flex h-full flex-col items-center">
      <div className="flex flex-1 items-center">
        <Logo />
      </div>
      <div className="mb-10 flex flex-col items-center gap-2">
        <Spinner className="size-6 text-primary" />
        <p className="text-sm text-muted-foreground">{displayMessage}</p>
        {latestLog && (
          <p
            aria-hidden="true"
            className="max-w-sm line-clamp-3 whitespace-pre-wrap break-all text-center font-mono text-xs text-foreground"
          >
            {latestLog}
          </p>
        )}
        <output className="sr-only" aria-live="polite" aria-atomic="true">
          {announcedLog}
        </output>
      </div>
    </div>
  );
}

function StoppedContent({ onStartServer }: { onStartServer: () => void }) {
  return (
    <>
      <Logo />
      <div className="mt-8 flex flex-col items-center gap-2">
        <p className="text-sm font-medium text-foreground">Server stopped</p>
        <div className="mt-4">
          <ActionButton onClick={onStartServer}>Start Server</ActionButton>
        </div>
      </div>
    </>
  );
}

function ErrorContent({
  error,
  logs,
  onRetry,
  onCopyDiagnostics,
}: {
  error: string | null;
  logs: string[];
  onRetry: () => void;
  onCopyDiagnostics: () => Promise<CopySupportDiagnosticsResult>;
}) {
  // Rendered in full. The startup timeout message embeds the backend's last 20
  // lines, so capping it here would drop the only actionable part of the screen.
  const displayedError = useMemo(
    () => (error ? redactDiagnosticsText(error) : null),
    [error],
  );
  const serverOutput = useMemo(
    () =>
      visibleLines(redactDiagnosticsText(logs.join("\n")))
        .split("\n")
        .slice(-20)
        .join("\n"),
    [logs],
  );
  const comparableError = displayedError ? visibleLines(displayedError) : null;

  return (
    <div className="flex max-h-full w-full flex-col items-center overflow-y-auto py-4">
      <Logo />
      <div className="mt-8 flex w-full flex-col items-center gap-2">
        <div
          role="alert"
          aria-live="assertive"
          aria-atomic="true"
          className="flex flex-col items-center gap-2"
        >
          <p className="text-sm font-medium text-destructive">Something went wrong</p>
          {displayedError && (
            <p className="max-w-md whitespace-pre-wrap break-words text-center text-xs text-muted-foreground">
              {displayedError}
            </p>
          )}
        </div>
        {serverOutput && !comparableError?.includes(serverOutput) && (
          <details className="mt-2 w-full max-w-md text-left text-xs text-muted-foreground">
            <summary className="cursor-pointer select-none text-center">Server output</summary>
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/40 bg-muted/30 p-3 font-mono text-ui-11 leading-relaxed">
              {serverOutput}
            </pre>
          </details>
        )}
        <DiagnosticsCopyActions onCopyDiagnostics={onCopyDiagnostics}>
          <ActionButton onClick={onRetry}>Retry</ActionButton>
        </DiagnosticsCopyActions>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function StartupScreen({
  status,
  logs,
  error,
  currentStepIndex,
  progressDetail,
  startupMessage,
  elevationPackages,
  onInstall,
  onRetry,
  onRetryInstall,
  onApproveElevation,
  onStartServer,
  onCopyDiagnostics,
}: StartupScreenProps) {
  function renderContent() {
    switch (status) {
      case "checking":
        return <CheckingContent />;
      case "not-installed":
        return <NotInstalledContent onInstall={onInstall} />;
      case "installing":
        return (
          <InstallingContent
            logs={logs}
            currentStepIndex={currentStepIndex}
            progressDetail={progressDetail}
          />
        );
      case "install-error":
        return (
          <InstallErrorContent
            error={error}
            onRetryInstall={onRetryInstall}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
      case "repairing":
        return <RepairingContent logs={logs} progressDetail={progressDetail} />;
      case "repair-error":
        return (
          <RepairErrorContent
            error={error}
            onRetry={onRetry}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
      case "needs-elevation":
        return (
          <NeedsElevationContent
            elevationPackages={elevationPackages}
            onApproveElevation={onApproveElevation}
            onRetryInstall={onRetryInstall}
          />
        );
      case "starting":
        return (
          <StartingContent
            key={startupMessage}
            message={startupMessage}
            logs={logs}
          />
        );
      case "running":
        return null;
      case "stopped":
        return <StoppedContent onStartServer={onStartServer} />;
      case "error":
        return (
          <ErrorContent
            error={error}
            logs={logs}
            onRetry={onRetry}
            onCopyDiagnostics={onCopyDiagnostics}
          />
        );
    }
  }

  return (
    <div className="box-border flex h-full w-full flex-col items-center overflow-y-auto bg-background pb-6 pt-[var(--studio-startup-top-inset,0px)]">
      <div className="flex min-h-0 flex-1 w-full max-w-md items-center justify-center px-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={status}
            className="flex h-full w-full flex-col items-center justify-center text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: EASE_OUT_QUART }}
          >
            {renderContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
