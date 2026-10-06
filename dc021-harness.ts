// CI-only A/B driver for the browser panel overlay states. Not part of the PR.
import { useBrowserStore } from "@/features/browser/store";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { toast } from "@/lib/toast";

const HELPER = "http://127.0.0.1:8765";
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const log = (m: string) => fetch(`${HELPER}/log?m=${encodeURIComponent(m)}`).catch(() => undefined);
const shot = (name: string) => fetch(`${HELPER}/shot?name=${name}`).catch(() => undefined);

async function waitFor(check: () => boolean, ms: number, label: string): Promise<boolean> {
  const end = Date.now() + ms;
  let next = Date.now() + 30_000;
  while (Date.now() < end) {
    if (check()) return true;
    if (Date.now() > next) {
      next = Date.now() + 30_000;
      await log(`waiting ${label} path=${location.pathname}`);
      await shot(`wait-${label}-${Date.now()}`);
    }
    await sleep(250);
  }
  await log(`TIMEOUT ${label} path=${location.pathname}`);
  await shot(`timeout-${label}`);
  return false;
}

function button(label: string): HTMLElement | null {
  return (
    [...document.querySelectorAll<HTMLElement>(`button[aria-label="${label}"]`)].find(
      (candidate) => candidate.offsetParent !== null,
    ) ?? null
  );
}

function pageState(): string {
  const el = document.querySelector<HTMLElement>("[data-native-page]");
  const bg = el?.style.background ? "snapshot" : "plain";
  const inset =
    getComputedStyle(document.documentElement).getPropertyValue("--studio-browser-page-inset") || "none";
  const r = el?.getBoundingClientRect();
  const toastBox = document.querySelector("[data-sonner-toast]")?.getBoundingClientRect();
  const tip = document.querySelector('[role="tooltip"]')?.parentElement?.getBoundingClientRect();
  const fmt = (b?: DOMRect) =>
    b ? `${Math.round(b.left)},${Math.round(b.top)} ${Math.round(b.width)}x${Math.round(b.height)}` : "none";
  return `placeholder=${bg} inset=${inset.trim()} page=${fmt(r)} toast=${fmt(toastBox)} tooltip=${fmt(tip)} win=${innerWidth}x${innerHeight}`;
}

async function step(name: string, act: () => void | Promise<void>, settle = 1200) {
  await act();
  await sleep(settle);
  await log(`${name}: ${pageState()}`);
  await shot(name);
}

async function run() {
  await log(`harness start path=${location.pathname}`);
  if (!(await waitFor(() => location.pathname.startsWith("/chat"), 420_000, "chat"))) return;
  try {
    // biome-ignore lint/suspicious/noExplicitAny: CI-only
    const tauri = (window as any).__TAURI__;
    const win = tauri?.window?.getCurrentWindow?.();
    await win?.setSize?.(new tauri.dpi.LogicalSize(1440, 860));
    await win?.center?.();
  } catch (e) {
    await log(`resize failed ${e}`);
  }
  await sleep(3000);
  const store = useBrowserStore.getState();
  store.splitWithChatOn("left");
  store.openUrl("https://example.com/");
  await waitFor(() => !!document.querySelector("[data-native-page]"), 30_000, "native");
  await sleep(8000);
  await step("1-page", () => undefined);

  await step("2-tooltip", () => button("Reload")?.focus(), 1500);
  (document.activeElement as HTMLElement | null)?.blur();
  await sleep(800);

  await step("3-toast-split", () => {
    toast("Plain toast (split view)", { duration: 20_000 });
  });
  toast.dismiss();
  await sleep(1500);

  await step(
    "4-dropdown",
    () => {
      const more = button("More");
      more?.focus();
      more?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    },
    1500,
  );
  document.activeElement?.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  await sleep(1500);

  await step("5-settings-dialog", () => useSettingsDialogStore.getState().openDialog(), 2000);
  useSettingsDialogStore.getState().closeDialog();
  await sleep(1500);
  await step("6-after-close", () => undefined);

  useBrowserStore.getState().setFullView(true);
  await sleep(2000);
  await step("7-toast-fullview", () => {
    toast("Plain toast (full view)", { duration: 20_000 });
  });
  toast.dismiss();
  await sleep(1500);
  await step("8-fullview-after-toast", () => undefined);
  await log("harness done");
  await fetch(`${HELPER}/done`).catch(() => undefined);
}

void run();
