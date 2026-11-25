// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { unloadModel } from "@/features/chat/api/chat-api";
import { confirmStopRunningChatsIfNeeded } from "@/features/chat/utils/confirm-stop-running-chats";
import { requestScopedLocalPromptQueueStop } from "@/features/chat/utils/prompt-queue-boundary";

/** /unload refuses while chats generate on the model unless forced, so ask first.
 *  False when the user keeps them running: nothing was unloaded. */
export async function unloadAfterChatStopConfirmation(
  modelPath: string,
): Promise<boolean> {
  const decision = await confirmStopRunningChatsIfNeeded(
    "Unloading this model",
    "unloadOne",
    modelPath,
  );
  if (!decision.proceed) return false;
  requestScopedLocalPromptQueueStop(decision.promptQueueThreadIds);
  await unloadModel({
    model_path: modelPath,
    force_cancel_active: decision.forceCancelActive,
  });
  return true;
}

export const UNLOAD_DECLINED = Symbol("unload declined");

/** For a read, unload, re-read loop: a decline ends it instead of reading as a no-op unload. */
export async function unloadOrAbortIfDeclined(
  modelPath: string,
): Promise<void> {
  if (!(await unloadAfterChatStopConfirmation(modelPath))) {
    throw UNLOAD_DECLINED;
  }
}
