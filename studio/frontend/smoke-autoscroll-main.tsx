// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_chat_autoscroll.py; vite entry, no backend.

import {
  IntentAwareScrollProvider,
  useIntentAwareAutoScroll,
} from "@/components/assistant-ui/use-intent-aware-autoscroll";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const PARAGRAPH =
  "British cultural exports carry several distinct threads, and the reception of each " +
  "varies by audience, decade and medium. This paragraph exists to give the viewport real " +
  "prose to lay out and to wrap over several lines at this width.";

/** A reply whose thinking block folds, as reasoning.tsx folds it: optional tool run under it, and an
 *  unrelated element that can be removed separately. */
type FoldScene = { foldPx: number; siblingPx: number; extraPx: number };

function Harness(): ReactElement {
  const { ref, context } = useIntentAwareAutoScroll();
  const [blocks, setBlocks] = useState<string[]>([]);
  const [tail, setTail] = useState("");
  const [scene, setScene] = useState<FoldScene | null>(null);
  const viewportRef = useRef<HTMLElement | null>(null);
  const spacerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const viewport = () => viewportRef.current;
    const api = {
      seed(count = 40): void {
        setBlocks(Array.from({ length: count }, (_, index) => `${index}`));
        setTail("");
      },
      token(text: string): void {
        setTail((current) => current + text);
      },
      block(): void {
        setBlocks((current) => [...current, `${current.length}`]);
        setTail("");
      },
      /** Grow content with no mutation or resize, like a late image or webfont. */
      growSilently(px: number): void {
        const spacer = spacerRef.current;
        if (spacer) spacer.style.height = `${px}px`;
      },
      resetGrowth(): void {
        const spacer = spacerRef.current;
        if (spacer) spacer.style.height = "0px";
      },
      distanceFromBottom(): number {
        const element = viewport();
        if (!element) return -1;
        return Math.max(
          0,
          element.scrollHeight - element.scrollTop - element.clientHeight,
        );
      },
      isAtBottom(): boolean {
        return context.getIsAtBottom();
      },
      scrollUpBy(px: number): void {
        const element = viewport();
        if (element) element.scrollTop = Math.max(0, element.scrollTop - px);
      },
      scrollDownBy(px: number): void {
        const element = viewport();
        if (element) element.scrollTop += px;
      },
      detach(): void {
        context.detachFromBottom();
      },
      scrollToBottom(): void {
        context.scrollToBottom("instant");
      },
      foldScene(blockCount: number, next: FoldScene): void {
        setBlocks(Array.from({ length: blockCount }, (_, index) => `${index}`));
        setTail("");
        setScene(next);
      },
      /** The block hands itself to the hold while still open, then folds in the same task, as a
       *  reduced-motion close does; its tool run hides with it. */
      fold(collapsedPx: number): void {
        const element = viewport();
        const block = element?.querySelector<HTMLElement>('[data-smoke="fold"]');
        if (!block) return;
        context.holdThroughCollapse();
        const body = block.firstElementChild as HTMLElement | null;
        if (body) body.style.height = `${collapsedPx}px`;
        element?.querySelector('[data-smoke="sibling"]')?.setAttribute("hidden", "");
      },
      removeExtra(): void {
        viewport()?.querySelector('[data-smoke="extra"]')?.setAttribute("hidden", "");
      },
      /** Viewport-relative tops: a fold that keeps the reader still leaves these where they were. */
      foldTops(): { block: number; tail: number; tailBottom: number; padding: number } {
        const element = viewport();
        const box = element?.getBoundingClientRect();
        const block = element?.querySelector('[data-smoke="fold"]')?.getBoundingClientRect();
        const reply = element?.querySelector('[data-smoke="reply-tail"]')?.getBoundingClientRect();
        return {
          block: block && box ? block.top - box.top : -1,
          tail: reply && box ? reply.top - box.top : -1,
          tailBottom: reply && box ? reply.bottom - box.top : -1,
          padding: element
            ? Number.parseFloat(element.style.getPropertyValue("--aui-scroll-stabilizer")) || 0
            : 0,
        };
      },
      metrics(): { scrollHeight: number; scrollTop: number; clientHeight: number } {
        const element = viewport();
        if (!element) return { scrollHeight: -1, scrollTop: -1, clientHeight: -1 };
        return {
          scrollHeight: element.scrollHeight,
          scrollTop: element.scrollTop,
          clientHeight: element.clientHeight,
        };
      },
    };
    (window as unknown as { __autoscroll: typeof api }).__autoscroll = api;
  }, [context]);

  return (
    <IntentAwareScrollProvider value={context}>
      <div
        data-smoke="viewport"
        ref={(element) => {
          viewportRef.current = element;
          ref(element);
        }}
        style={{
          height: "100vh",
          width: "760px",
          overflowY: "auto",
          padding: "16px",
          paddingBottom: "var(--aui-scroll-stabilizer, 0px)",
          // As the thread viewport: a block container leaves its bottom padding out of scrollHeight.
          display: "flex",
          flexDirection: "column",
        }}
      >
        {blocks.map((id) => (
          <p key={id} style={{ margin: "0 0 16px" }}>
            Message {id}. {PARAGRAPH}
          </p>
        ))}
        {scene ? (
          <div data-role="assistant" data-smoke="reply">
            {scene.extraPx > 0 ? (
              <div data-smoke="extra" style={{ height: `${scene.extraPx}px` }} />
            ) : null}
            <div data-slot="reasoning-root" data-smoke="fold">
              <div style={{ height: `${scene.foldPx}px` }} />
            </div>
            {scene.siblingPx > 0 ? (
              <div data-slot="tool-run-under-thinking" data-smoke="sibling" style={{ height: `${scene.siblingPx}px` }} />
            ) : null}
            <p data-smoke="reply-tail" style={{ margin: "0 0 16px" }}>
              {tail || "The answer starts here."}
            </p>
          </div>
        ) : (
          <p data-smoke="tail" style={{ margin: "0 0 16px" }}>
            {tail}
          </p>
        )}
        {/* No flex shrink: the column would squeeze it to nothing and silent growth would not grow. */}
        <div ref={spacerRef} data-smoke="spacer" style={{ height: "0px", flexShrink: 0 }} />
      </div>
    </IntentAwareScrollProvider>
  );
}

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}
createRoot(root).render(<Harness />);
