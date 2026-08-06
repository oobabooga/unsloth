// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const ESCAPE_CHARACTER = String.fromCharCode(27);
const OSC_ESCAPE_PATTERN = new RegExp(
  `${ESCAPE_CHARACTER}\\][^\n\r]*?(?:${String.fromCharCode(7)}|${ESCAPE_CHARACTER}\\\\)`,
  "g",
);
const UNTERMINATED_OSC_ESCAPE_PATTERN = new RegExp(
  `${ESCAPE_CHARACTER}\\][^\n\r]*$`,
  "gm",
);
const ANSI_ESCAPE_PATTERN = new RegExp(
  `${ESCAPE_CHARACTER}(?:[@-Z\\-_]|\\[[0-?]*[ -/]*[@-~])`,
  "g",
);
const PRIVATE_KEY_BEGIN_PATTERN = /-----BEGIN [^-]*PRIVATE KEY-----/i;
const PRIVATE_KEY_END_PATTERN = /-----END [^-]*PRIVATE KEY-----/i;

function stripTerminalControlSequences(text: string): string {
  return text
    .replace(OSC_ESCAPE_PATTERN, "")
    .replace(UNTERMINATED_OSC_ESCAPE_PATTERN, "")
    .replace(ANSI_ESCAPE_PATTERN, "");
}

export function redactDiagnosticsText(text: string): string {
  let redacted = stripTerminalControlSequences(text);

  redacted = redacted.replace(
    /-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----/gi,
    "<redacted-private-key>",
  );
  redacted = redacted.replace(
    /-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*$/gi,
    "<redacted-private-key>",
  );
  redacted = redacted.replace(
    /\b[a-z][a-z0-9+.-]{0,31}:\/\/[^/\s]+/gi,
    (authority) => {
      const authorityStart = authority.indexOf("://") + 3;
      const credentialEnd = authority.lastIndexOf("@");
      if (credentialEnd < authorityStart) return authority;
      return `${authority.slice(0, authorityStart)}<redacted>@${authority.slice(credentialEnd + 1)}`;
    },
  );
  redacted = redacted.replace(
    /([?&][^=\s&`]+)=[^&#\s`]+/g,
    "$1=<redacted>",
  );
  redacted = redacted.replace(
    /(https?:\/\/[^\s`#]+)#[^\s`]+/gi,
    "$1#<redacted>",
  );
  redacted = redacted.replace(
    /\b(authorization\s*[:=]\s*)(bearer|basic)\s+[^\s,;]+/gi,
    "$1$2 <redacted>",
  );
  redacted = redacted.replace(/\bhf_[A-Za-z0-9]{20,}\b/g, "hf_<redacted>");
  redacted = redacted.replace(/\bghp_[A-Za-z0-9_]{20,}\b/g, "ghp_<redacted>");
  redacted = redacted.replace(/\bgithub_pat_[A-Za-z0-9_]{20,}\b/g, "github_pat_<redacted>");
  redacted = redacted.replace(/\bsk-[A-Za-z0-9_-]{20,}\b/g, "sk-<redacted>");
  redacted = redacted.replace(
    /\b(cookie|set-cookie)\s*[:=]\s*[^\n\r]+/gi,
    "$1=<redacted>",
  );
  redacted = redacted.replace(
    /(^|[\s;])((?:[A-Z0-9]+_)*(?:TOKEN|KEY|SECRET|PASSWORD)(?:_[A-Z0-9]+)*\s*=\s*)[^\s]+/gi,
    "$1$2<redacted>",
  );
  redacted = redacted.replace(
    /(\b(?:native_path_lease|nativePathLease)["']?\s*[:=]\s*["']?)[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/gi,
    "$1<redacted-native-path-lease>",
  );

  // Redact Unsloth paths before broader home-directory paths.
  redacted = redacted.replace(
    /(?:\/Users|\/home)\/[^\s/]+\/\.unsloth\/studio/gi,
    "<studio_home>",
  );
  redacted = redacted.replace(
    /[A-Z]:\\Users\\[^\s\\]+\\\.unsloth\\studio/gi,
    "<studio_home>",
  );
  redacted = redacted.replace(/(?:\/Users|\/home)\/[^\s/]+/gi, "$HOME");
  redacted = redacted.replace(/[A-Z]:\\Users\\[^\s\\]+/gi, "%USERPROFILE%");
  redacted = redacted.replace(
    /\b[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,63}\b/gi,
    "<redacted-email>",
  );

  return redacted;
}

interface RedactedLogLine {
  redacted: string;
  beginsPrivateKey: boolean;
  endsPrivateKey: boolean;
}

interface RedactedStreamLine {
  raw: string;
  displayed: string | null;
}

export interface DiagnosticsLineStreamState {
  rawLines: string[];
  lines: RedactedStreamLine[];
  displayedLines: string[];
  insidePrivateKey: boolean;
}

// The startup screen rescans its whole log buffer on every server-log event, so
// without this the ~15 redaction passes would run over up to 500 lines per line
// the backend prints. Log lines repeat rarely, but a line is re-examined once per
// subsequent event, which is exactly what this cache collapses to a lookup.
const REDACTED_LINE_CACHE_LIMIT = 2_000;
const redactedLineCache = new Map<string, RedactedLogLine>();

function redactLogLine(rawLine: string): RedactedLogLine {
  const cached = redactedLineCache.get(rawLine);
  if (cached) return cached;

  const sanitizedLine = stripTerminalControlSequences(rawLine);
  const entry: RedactedLogLine = {
    redacted: redactDiagnosticsText(sanitizedLine),
    beginsPrivateKey: PRIVATE_KEY_BEGIN_PATTERN.test(sanitizedLine),
    endsPrivateKey: PRIVATE_KEY_END_PATTERN.test(sanitizedLine),
  };

  if (redactedLineCache.size >= REDACTED_LINE_CACHE_LIMIT) redactedLineCache.clear();
  redactedLineCache.set(rawLine, entry);
  return entry;
}

function appendRedactedStreamLine(
  rawLine: string,
  insidePrivateKey: boolean,
): { line: RedactedStreamLine; insidePrivateKey: boolean } {
  const { redacted, beginsPrivateKey, endsPrivateKey } = redactLogLine(rawLine);

  if (insidePrivateKey) {
    return {
      line: { raw: rawLine, displayed: null },
      insidePrivateKey: !endsPrivateKey,
    };
  }

  return {
    line: { raw: rawLine, displayed: redacted },
    insidePrivateKey: beginsPrivateKey && !endsPrivateKey,
  };
}

export interface DiagnosticsLineRedactor {
  redactLine: (rawLine: string) => string | null;
  reset: () => void;
}

/**
 * Redacts child output before it enters a bounded retained-log buffer. Keeping
 * private-key state here prevents a body line from becoming visible after its
 * opening marker has already fallen out of that buffer or the UI remounts.
 */
export function createDiagnosticsLineRedactor(): DiagnosticsLineRedactor {
  let insidePrivateKey = false;
  return {
    redactLine(rawLine) {
      const next = appendRedactedStreamLine(rawLine, insidePrivateKey);
      insidePrivateKey = next.insidePrivateKey;
      return next.line.displayed;
    },
    reset() {
      insidePrivateKey = false;
    },
  };
}

function sameLinesAt(
  left: readonly string[],
  leftStart: number,
  right: readonly string[],
  rightStart: number,
  count: number,
): boolean {
  for (let index = 0; index < count; index += 1) {
    if (left[leftStart + index] !== right[rightStart + index]) return false;
  }
  return true;
}

/**
 * Redacts an append-only bounded log stream without re-running the redaction
 * patterns over every retained line for each child-output event. The common
 * full-buffer case drops one old line and appends one new line; retained
 * redactions and private-key state stay intact across that shift.
 */
export function redactDiagnosticsLineStream(
  rawLines: readonly string[],
  previous?: DiagnosticsLineStreamState,
): DiagnosticsLineStreamState {
  const nextRawLines = Array.from(rawLines);
  let lines: RedactedStreamLine[] = [];
  let firstNewLine = 0;
  let insidePrivateKey = false;

  if (
    previous &&
    previous.rawLines.length <= nextRawLines.length &&
    sameLinesAt(
      previous.rawLines,
      0,
      nextRawLines,
      0,
      previous.rawLines.length,
    )
  ) {
    lines = previous.lines.slice();
    firstNewLine = previous.rawLines.length;
    insidePrivateKey = previous.insidePrivateKey;
  } else if (
    previous &&
    previous.rawLines.length === nextRawLines.length &&
    nextRawLines.length > 0 &&
    sameLinesAt(
      previous.rawLines,
      1,
      nextRawLines,
      0,
      nextRawLines.length - 1,
    )
  ) {
    lines = previous.lines.slice(1);
    firstNewLine = nextRawLines.length - 1;
    insidePrivateKey = previous.insidePrivateKey;
  }

  for (let index = firstNewLine; index < nextRawLines.length; index += 1) {
    const next = appendRedactedStreamLine(
      nextRawLines[index],
      insidePrivateKey,
    );
    lines.push(next.line);
    insidePrivateKey = next.insidePrivateKey;
  }

  return {
    rawLines: nextRawLines,
    lines,
    displayedLines: lines.flatMap((line) =>
      line.displayed === null ? [] : [line.displayed],
    ),
    insidePrivateKey,
  };
}

export function latestRedactedDiagnosticsLine(lines: string[]): string {
  let insidePrivateKey = false;
  let latestVisibleLine = "";

  for (const rawLine of lines) {
    const { redacted, beginsPrivateKey, endsPrivateKey } = redactLogLine(rawLine);

    if (insidePrivateKey) {
      if (endsPrivateKey) insidePrivateKey = false;
      continue;
    }

    if (redacted.trim()) latestVisibleLine = redacted;
    if (beginsPrivateKey && !endsPrivateKey) insidePrivateKey = true;
  }

  return latestVisibleLine;
}
