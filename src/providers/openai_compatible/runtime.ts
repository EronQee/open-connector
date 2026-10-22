import type { CredentialValidationResult } from "../../core/types.ts";
import type { ProviderActionHandlers } from "../provider-runtime.ts";

import {
  base64Bytes,
  compactObject,
  optionalInteger,
  optionalNumber,
  optionalRecord,
  optionalString,
  optionalStringArray,
} from "../../core/cast.ts";
import {
  assertPublicHttpUrl,
  encodePathSegment,
  isPrivateNetworkAccessAllowed,
  readBoundedResponseBytes,
} from "../../core/request.ts";
import {
  ProviderRequestError,
  defaultProviderJsonMaxResponseBytes,
  providerFetch,
  providerUserAgent,
  readProviderErrorTextBody,
  readProviderJson,
  requiredInputString,
  requiredResponseRecord,
  runProviderRequest,
} from "../provider-runtime.ts";

/**
 * Text models can stream-token for minutes and image generation routinely
 * takes longer than the 30 second provider default, so these actions carry
 * explicit budgets.
 */
const chatCompletionTimeoutMs = 120_000;
const responseTimeoutMs = 300_000;
const imageGenerationTimeoutMs = 300_000;

/**
 * Async image relays accept the task and answer in seconds; polling one task
 * is a cheap read. Both stay well under the generation budget.
 */
const imageTaskSubmitTimeoutMs = 120_000;
const imageTaskQueryTimeoutMs = 60_000;

/** The async submit route relays document; `submitPath` overrides it. */
const defaultImageTaskSubmitPath = "images/generations/async";

/**
 * Relays disagree on where the task record lives after an async submit.
 * Querying is idempotent, so the first path that is not a routing 404 wins.
 */
const imageTaskQueryPaths = ["tasks", "images/tasks", "images/generations"];

/** Reference images are downloaded to build a multipart edit request. */
const referenceImageMaxBytes = 20 * 1024 * 1024;

const imageApiLabel = "OpenAI-compatible API";

type OpenAiCompatibleActionHandler = (
  input: Record<string, unknown>,
  context: OpenAiCompatibleActionContext,
) => Promise<unknown>;

export interface OpenAiCompatibleActionContext {
  baseUrl: string;
  apiKey: string;
  fetcher: typeof fetch;
  signal?: AbortSignal;
}

interface OpenAiCompatibleJsonRequest {
  context: OpenAiCompatibleActionContext;
  method: "GET" | "POST";
  path: string;
  body?: Record<string, unknown> | FormData;
  /**
   * Body to retry once with when the multipart form is rejected for a reason
   * other than authentication, such as a relay that only parses JSON.
   */
  fallbackBody?: Record<string, unknown>;
  timeoutMs?: number;
  phase: "validate" | "execute";
}

interface NormalizedChatUsage {
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
}

interface NormalizedImageItem {
  url: string | null;
  b64Json: string | null;
  revisedPrompt: string | null;
}

export const openAiCompatibleActionHandlers: ProviderActionHandlers<
  "openai_compatible",
  OpenAiCompatibleActionHandler
> = {
  list_models(_input, context) {
    return listModels(context);
  },
  create_chat_completion(input, context) {
    return createChatCompletion(input, context);
  },
  create_response(input, context) {
    return createResponse(input, context);
  },
  create_image(input, context) {
    return createImage(input, context);
  },
  edit_image(input, context) {
    return editImage(input, context);
  },
  submit_image_task(input, context) {
    return submitImageTask(input, context);
  },
  get_image_task(input, context) {
    return getImageTask(input, context);
  },
};

export async function validateOpenAiCompatibleCredential(
  input: Record<string, string>,
  fetcher: typeof fetch,
  signal?: AbortSignal,
): Promise<CredentialValidationResult> {
  const context = createOpenAiCompatibleContext(input, fetcher, signal);
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "GET",
    path: "models",
    phase: "validate",
  });

  return {
    profile: {
      accountId: new URL(context.baseUrl).host,
      displayName: `OpenAI-compatible API @ ${new URL(context.baseUrl).host}`,
    },
    grantedScopes: [],
    metadata: {
      baseUrl: context.baseUrl,
      availableModels: readModelIds(payload),
    },
  };
}

export function createOpenAiCompatibleContext(
  input: Record<string, string>,
  fetcher: typeof fetch,
  signal?: AbortSignal,
): OpenAiCompatibleActionContext {
  return {
    baseUrl: normalizeOpenAiCompatibleBaseUrl(input.baseUrl),
    apiKey: requiredInputString(input.apiKey, "apiKey"),
    fetcher,
    signal,
  };
}

/**
 * Normalize a relay base URL to the versioned API root every request path
 * joins against. A bare host gets `/v1/` appended because relay services
 * document their base either way, and the trailing slash keeps relative
 * `new URL(path, base)` joins inside the version segment.
 */
export function normalizeOpenAiCompatibleBaseUrl(
  value: unknown,
  allowPrivateNetwork: boolean = isPrivateNetworkAccessAllowed(),
): string {
  const raw = requiredInputString(value, "baseUrl");
  const url = assertPublicHttpUrl(raw, {
    fieldName: "baseUrl",
    createError: (message) => new ProviderRequestError(400, message),
    allowPrivateNetwork,
  });
  const trimmedPath = url.pathname.replace(/\/+$/, "");
  url.pathname = trimmedPath === "" ? "/v1/" : `${trimmedPath}/`;
  url.hash = "";
  return url.toString();
}

async function listModels(context: OpenAiCompatibleActionContext): Promise<unknown> {
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "GET",
    path: "models",
    phase: "execute",
  });
  const models = readModelRecords(payload);

  return {
    models,
    total: models.length,
    raw: payload,
  };
}

async function createChatCompletion(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const messages = readChatMessages(input.messages);
  const body = compactObject({
    model: requiredInputString(input.model, "model"),
    messages,
    temperature: optionalNumber(input.temperature),
    top_p: optionalNumber(input.topP),
    max_tokens: optionalInteger(input.maxTokens),
    stop: optionalString(input.stop) ?? optionalStringArray(input.stop),
  });
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: "chat/completions",
    body,
    timeoutMs: chatCompletionTimeoutMs,
    phase: "execute",
  });

  return normalizeChatCompletion(payload);
}

async function createResponse(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const body = compactObject({
    model: requiredInputString(input.model, "model"),
    input: input.input,
    instructions: optionalString(input.instructions),
    max_output_tokens: optionalInteger(input.maxOutputTokens),
    previous_response_id: optionalString(input.previousResponseId),
    store: typeof input.store === "boolean" ? input.store : undefined,
  });
  if (body.input === undefined) {
    throw new ProviderRequestError(400, "input is required");
  }
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: "responses",
    body,
    timeoutMs: responseTimeoutMs,
    phase: "execute",
  });

  return {
    id: optionalString(payload.id) ?? null,
    status: optionalString(payload.status) ?? null,
    outputText: readResponsesOutputText(payload),
    usage: readUsage(payload.usage),
    raw: payload,
  };
}

async function createImage(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const body = mergeExtraBody(
    buildImageRequestBody(input, {
      model: requiredInputString(input.model, "model"),
      prompt: requiredInputString(input.prompt, "prompt"),
    }),
    input,
  );
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: "images/generations",
    body,
    timeoutMs: imageGenerationTimeoutMs,
    phase: "execute",
  });

  return normalizeImageResponse(payload);
}

/**
 * Edit an image with the relay's image edits route. The official OpenAI
 * contract is multipart, but many relays only accept a JSON body with
 * reference images inline. Multipart goes first because that is the documented
 * shape; a routing rejection falls back to JSON rather than reporting a 400
 * the relay never meant.
 */
async function editImage(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const model = requiredInputString(input.model, "model");
  const prompt = requiredInputString(input.prompt, "prompt");
  const images = readReferenceImageValues(input.image);
  if (images.length === 0) {
    throw new ProviderRequestError(400, "image is required");
  }
  const mask = optionalString(input.mask);
  const body = mergeExtraBody(buildImageRequestBody(input, { model, prompt }), input);

  const formData = await buildEditFormData(body, images, mask, context);
  const jsonBody: Record<string, unknown> = {
    ...body,
    image: images.map(toInlineImageReference),
    mask: mask === undefined ? undefined : toInlineImageReference(mask),
  };
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: "images/edits",
    body: formData ?? jsonBody,
    timeoutMs: imageGenerationTimeoutMs,
    phase: "execute",
    // Only a multipart body is worth retrying; when the form could not be
    // built the JSON body is already what was sent.
    fallbackBody: formData ? jsonBody : undefined,
  });

  return normalizeImageResponse(payload);
}

/**
 * Build the multipart body, or `undefined` when a reference image cannot be
 * downloaded. Relays that only parse JSON fetch the URL themselves, so an
 * unreachable reference is their call to make; an invalid URL is still the
 * caller's and is rethrown.
 */
async function buildEditFormData(
  body: Record<string, unknown>,
  images: string[],
  mask: string | undefined,
  context: OpenAiCompatibleActionContext,
): Promise<FormData | undefined> {
  const formData = new FormData();
  for (const [key, value] of Object.entries(body)) {
    if (value === undefined) {
      continue;
    }
    formData.append(key, typeof value === "string" ? value : JSON.stringify(value));
  }
  try {
    for (const image of images) {
      formData.append("image[]", await imageInputToBlob(image, "image", context));
    }
    if (mask) {
      formData.append("mask", await imageInputToBlob(mask, "mask", context));
    }
  } catch (error) {
    if (error instanceof ProviderRequestError && error.status === 400) {
      throw error;
    }
    return undefined;
  }
  return formData;
}

function readReferenceImageValues(value: unknown): string[] {
  const single = optionalString(value);
  if (single) {
    return [single];
  }
  return optionalStringArray(value) ?? [];
}

/**
 * The JSON fallback carries base64 image data without the `data:` prefix, which
 * is the shape most OpenAI-compatible relays parse.
 */
function toInlineImageReference(value: string): string {
  return value.startsWith("data:") ? (value.split(",").at(1) ?? value) : value;
}

/**
 * Multipart form parts need bytes, so URLs are downloaded through the guarded
 * fetcher and data URLs are decoded locally. A bare base64 payload is accepted
 * as-is because some relays document that form directly.
 */
async function imageInputToBlob(
  value: string,
  fieldName: string,
  context: OpenAiCompatibleActionContext,
): Promise<Blob> {
  const dataUrl = parseDataUrl(value);
  if (dataUrl) {
    return dataUrl.isBase64
      ? new Blob([decodeImageBase64(dataUrl.data, fieldName)], { type: dataUrl.mimeType })
      : new Blob([decodeURIComponent(dataUrl.data)], { type: dataUrl.mimeType });
  }
  if (!looksLikeHttpUrl(value)) {
    // A bare base64 payload carries no declared type; image/png keeps the part
    // readable for relays that trust the declared content type.
    return new Blob([decodeImageBase64(value, fieldName)], { type: "image/png" });
  }
  const url = assertPublicHttpUrl(value, {
    fieldName,
    createError: (message) => new ProviderRequestError(400, message),
  });
  return runProviderRequest(
    { signal: context.signal, label: "OpenAI-compatible reference image download" },
    async (signal) => {
      const response = await providerFetch(url.toString(), {
        headers: { accept: "image/*", "user-agent": providerUserAgent },
        signal,
      });
      if (!response.ok) {
        const text = await readProviderErrorTextBody(response, "reference image download error response");
        throw new ProviderRequestError(response.status >= 500 ? 502 : 400, text || `${fieldName} download failed`);
      }
      const bytes = await readBoundedResponseBytes(response, {
        maxBytes: referenceImageMaxBytes,
        fieldName,
        createError: (message) => new ProviderRequestError(413, message),
      });
      return new Blob([Uint8Array.from(bytes)], {
        type: response.headers.get("content-type") ?? "image/png",
      });
    },
  );
}

function decodeImageBase64(value: string, fieldName: string): Uint8Array<ArrayBuffer> {
  return base64Bytes(value, fieldName, (message) => new ProviderRequestError(400, message));
}

function parseDataUrl(value: string): { mimeType: string; data: string; isBase64: boolean } | undefined {
  const match = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(value);
  if (!match) {
    return undefined;
  }
  return { mimeType: match[1] || "image/png", data: match[3] ?? "", isBase64: Boolean(match[2]) };
}

function looksLikeHttpUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

async function submitImageTask(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const submitPath = optionalString(input.submitPath) ?? defaultImageTaskSubmitPath;
  const body = mergeExtraBody(
    buildImageRequestBody(input, {
      model: requiredInputString(input.model, "model"),
      prompt: requiredInputString(input.prompt, "prompt"),
    }),
    input,
  );
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: submitPath,
    body,
    timeoutMs: imageTaskSubmitTimeoutMs,
    phase: "execute",
  });
  const task = normalizeImageTask(payload);

  return {
    ...task,
    submitPath,
    raw: payload,
  };
}

async function getImageTask(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const taskId = requiredInputString(input.taskId, "taskId");
  const { payload, queryPath } = await readImageTaskPayload(taskId, input.queryPath, context);
  const task = normalizeImageTask(payload, taskId);

  return {
    ...task,
    taskId: task.taskId ?? taskId,
    queryPath,
    raw: payload,
  };
}

/**
 * Query the task record across the paths relays put it under. A 404 means
 * that path is not this relay's route, so the next candidate is tried; every
 * other failure is the relay answering about the task and is reported as-is.
 */
async function readImageTaskPayload(
  taskId: string,
  rawQueryPath: unknown,
  context: OpenAiCompatibleActionContext,
): Promise<{ payload: Record<string, unknown>; queryPath: string }> {
  const overridden = optionalString(rawQueryPath);
  const candidates = overridden ? [overridden] : imageTaskQueryPaths;
  let lastNotFound: ProviderRequestError | undefined;

  for (const candidate of candidates) {
    const queryPath = `${trimTrailingSlash(candidate)}/${encodePathSegment(taskId)}`;
    try {
      const payload = await requestOpenAiCompatibleJson({
        context,
        method: "GET",
        path: queryPath,
        timeoutMs: imageTaskQueryTimeoutMs,
        phase: "execute",
      });
      return { payload, queryPath };
    } catch (error) {
      if (error instanceof ProviderRequestError && error.status === 404) {
        lastNotFound = error;
        continue;
      }
      throw error;
    }
  }

  throw lastNotFound ?? new ProviderRequestError(502, "No image task query path answered for this relay.");
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

/**
 * Merge `extraBody` last so a caller can send relay-specific fields such as
 * `extra_body` or `resolution` without the connector modeling each one.
 */
function mergeExtraBody(body: Record<string, unknown>, input: Record<string, unknown>): Record<string, unknown> {
  const extraBody = optionalRecord(input.extraBody);
  return extraBody ? { ...body, ...extraBody } : body;
}

function buildImageRequestBody(input: Record<string, unknown>, base: Record<string, unknown>): Record<string, unknown> {
  return compactObject({
    ...base,
    n: optionalInteger(input.n),
    size: optionalString(input.size),
    quality: optionalString(input.quality),
    style: optionalString(input.style),
    background: optionalString(input.background),
    output_format: optionalString(input.outputFormat),
    response_format: optionalString(input.responseFormat),
    watermark: typeof input.watermark === "boolean" ? input.watermark : undefined,
  });
}

function normalizeChatCompletion(payload: Record<string, unknown>) {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const firstChoice = optionalRecord(choices[0]);
  const message = optionalRecord(firstChoice?.message);

  return {
    id: optionalString(payload.id) ?? null,
    model: optionalString(payload.model) ?? null,
    message: {
      role: optionalString(message?.role) ?? null,
      content: message?.content ?? null,
    },
    finishReason: optionalString(firstChoice?.finish_reason) ?? null,
    usage: readUsage(payload.usage),
    raw: payload,
  };
}

function normalizeImageResponse(payload: Record<string, unknown>) {
  return {
    created: optionalInteger(payload.created) ?? null,
    images: collectImageItems(payload),
    raw: payload,
  };
}

const imageUrlKeys = [
  "url",
  "image_url",
  "imageUrl",
  "output_url",
  "outputUrl",
  "result_url",
  "resultUrl",
  "download_url",
  "downloadUrl",
  "asset_url",
  "assetUrl",
];

const imageContainerKeys = [
  "data",
  "images",
  "result",
  "results",
  "output",
  "outputs",
  "content",
  "items",
  "files",
  "candidates",
  "parts",
  "task",
];

const imageBase64Keys = ["b64_json", "base64", "image_base64", "imageBase64"];

/**
 * Relays answer image generation in more shapes than the OpenAI envelope:
 * `{data:[{url}]}`, `{data:{result:{images:[...]}}}`, Gemini
 * `candidates[].content.parts[].inlineData`, an `image_generation_call` item,
 * and Markdown-embedded data URLs. Collecting from all of them turns a
 * successful generation into a readable result instead of a 502.
 */
function collectImageItems(payload: Record<string, unknown>): NormalizedImageItem[] {
  const found: NormalizedImageItem[] = [];

  const visit = (value: unknown, depth = 0): void => {
    if (depth > 8 || value === null || value === undefined) {
      return;
    }
    if (typeof value === "string") {
      found.push(...imagesInText(value));
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        visit(item, depth + 1);
      }
      return;
    }
    const record = optionalRecord(value);
    if (!record) {
      return;
    }

    // Dedupe inside one relay record only: a record that repeats its own
    // payload (a url field plus the same value under a base64 key, or the same
    // data URL in two fields) is one image, while two records that happen to
    // carry an identical value are two outputs and must both survive.
    const seen = new Set<string>();
    const add = (item: NormalizedImageItem) => {
      const key = item.url ?? item.b64Json;
      if (!key || seen.has(key)) {
        return;
      }
      seen.add(key);
      found.push(item);
    };

    const revisedPrompt = optionalString(record.revised_prompt) ?? optionalString(record.revisedPrompt) ?? null;

    if (record.type === "image_generation_call") {
      const result = optionalString(record.result);
      if (result) {
        add(
          looksLikeHttpUrl(result)
            ? { url: result, b64Json: null, revisedPrompt: null }
            : { url: null, b64Json: result, revisedPrompt: null },
        );
      }
    }

    // Pass 1: caller-fetchable URLs. A record that offers one is answered with
    // the URL, so the inline base64 copy of the same image is not transferred.
    let hasUrl = false;
    for (const key of imageUrlKeys) {
      const candidate = record[key];
      if (typeof candidate === "string" && candidate) {
        if (looksLikeHttpUrl(candidate) || candidate.startsWith("/")) {
          hasUrl = true;
          add({ url: candidate, b64Json: null, revisedPrompt });
        } else {
          for (const item of imagesInText(candidate)) {
            if (item.url) {
              hasUrl = true;
            }
            add(item);
          }
        }
      } else if (candidate !== undefined) {
        visit(candidate, depth + 1);
      }
    }
    // A content part keeps its payload in `text`, including the data URL some
    // relays paste into their answer instead of returning image bytes.
    const partText = optionalString(record.text);
    if (partText) {
      for (const item of imagesInText(partText)) {
        if (item.url) {
          hasUrl = true;
        }
        add(item);
      }
    }

    if (!hasUrl) {
      // Gemini-native parts carry the bytes under inlineData/inline_data, which
      // is not one of the container keys below.
      const inline = optionalRecord(record.inlineData) ?? optionalRecord(record.inline_data);
      const inlineData = optionalString(inline?.data);
      if (inlineData) {
        add({ url: null, b64Json: inlineData, revisedPrompt: null });
      }
      for (const key of imageBase64Keys) {
        const candidate = optionalString(record[key]);
        if (candidate) {
          add({ url: null, b64Json: candidate, revisedPrompt });
        }
      }
    }

    for (const key of imageContainerKeys) {
      if (record[key] !== undefined) {
        visit(record[key], depth + 1);
      }
    }
  };

  visit(payload);

  return found.map((item) => ({
    url: item.url,
    b64Json: item.b64Json,
    revisedPrompt: item.revisedPrompt,
  }));
}

/**
 * Pull image payloads out of text: a bare data URL, a Markdown image, or a
 * JSON-in-string body some relays wrap their result in.
 */
function imagesInText(text: string): NormalizedImageItem[] {
  const items: NormalizedImageItem[] = [];
  const dataUrl = /data:image\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+/gi;
  for (const match of text.matchAll(dataUrl)) {
    items.push({ url: null, b64Json: match[0].split(",").at(1) ?? null, revisedPrompt: null });
  }
  const markdown = /!\[[^\]]*\]\((https?:\/\/[^\s)]+)\)/g;
  for (const match of text.matchAll(markdown)) {
    items.push({ url: match[1] ?? null, b64Json: null, revisedPrompt: null });
  }
  return items;
}

const imageTaskStatusInProgress = new Set([
  "queued",
  "pending",
  "running",
  "processing",
  "in_progress",
  "in-progress",
  "generating",
  "submitted",
  "created",
]);

const imageTaskStatusCompleted = new Set([
  "success",
  "successful",
  "succeed",
  "succeeded",
  "completed",
  "complete",
  "done",
  "finished",
  "ok",
  "ready",
]);

const imageTaskStatusFailed = new Set([
  "failure",
  "failed",
  "fail",
  "error",
  "errored",
  "canceled",
  "cancelled",
  "timeout",
  "timedout",
  "rejected",
  "expired",
]);

export type NormalizedImageTaskState = "queued" | "in_progress" | "completed" | "failed" | "unknown";

interface NormalizedImageTask {
  taskId: string | null;
  status: string | null;
  state: NormalizedImageTaskState;
  progress: number | null;
  images: NormalizedImageItem[];
  error: { code: string | null; message: string | null } | null;
}

/**
 * Normalize an async image task. Relays answer failures with HTTP 200, so the
 * caller branches on `state`, not on the call succeeding.
 */
function normalizeImageTask(payload: Record<string, unknown>, knownTaskId?: string): NormalizedImageTask {
  const task =
    optionalRecord(payload.task) ??
    optionalRecord(payload.data) ??
    optionalRecord(optionalRecord(payload.data)?.task) ??
    payload;
  const status =
    optionalString(task.status) ?? optionalString(task.task_status) ?? optionalString(payload.status) ?? null;
  const images = collectImageItems(payload);

  return {
    taskId:
      knownTaskId ??
      optionalString(task.id) ??
      optionalString(task.task_id) ??
      optionalString(task.taskId) ??
      optionalString(payload.id) ??
      optionalString(payload.task_id) ??
      null,
    status,
    state: normalizeImageTaskState(status, images.length > 0),
    progress: optionalNumber(task.progress) ?? optionalNumber(payload.progress) ?? null,
    images,
    error: readImageTaskError(task) ?? readImageTaskError(payload),
  };
}

/**
 * A relay that returns image bytes has finished even when its status field is
 * missing or non-standard, which is why completed wins over the status table
 * once images are present.
 */
function normalizeImageTaskState(status: string | null, hasImages: boolean): NormalizedImageTaskState {
  const key = status?.trim().toLowerCase() ?? "";
  if (key && imageTaskStatusFailed.has(key)) {
    return "failed";
  }
  if (key && imageTaskStatusCompleted.has(key)) {
    return "completed";
  }
  if (hasImages) {
    return "completed";
  }
  if (!key) {
    return "unknown";
  }
  if (key === "queued") {
    return "queued";
  }
  if (imageTaskStatusInProgress.has(key)) {
    return "in_progress";
  }
  return "unknown";
}

function readImageTaskError(value: Record<string, unknown>): { code: string | null; message: string | null } | null {
  const error = optionalRecord(value.error);
  const message =
    optionalString(error?.message) ?? optionalString(value.fail_reason) ?? optionalString(value.message) ?? null;
  const code = optionalString(error?.code) ?? optionalString(value.fail_code) ?? null;
  if (!message && !code) {
    return null;
  }
  return { code, message };
}

function readChatMessages(value: unknown): Array<{ role: string; content: unknown }> {
  const items = Array.isArray(value) ? value : [];
  const messages = items.flatMap((item) => {
    const record = optionalRecord(item);
    if (!record) {
      return [];
    }
    const role = requiredInputString(record.role, "messages[].role");
    return [{ role, content: record.content ?? null }];
  });
  if (messages.length === 0) {
    throw new ProviderRequestError(400, "messages is required");
  }
  return messages;
}

function readResponsesOutputText(payload: Record<string, unknown>): string | null {
  const direct = optionalString(payload.output_text);
  if (direct !== undefined) {
    return direct;
  }
  const outputItems = Array.isArray(payload.output) ? payload.output : [];
  const parts = outputItems.flatMap((item) => {
    const record = optionalRecord(item);
    if (!record || record.type !== "message") {
      return [];
    }
    const contentItems = Array.isArray(record.content) ? record.content : [];
    return contentItems.flatMap((content) => {
      const contentRecord = optionalRecord(content);
      const text = contentRecord?.type === "output_text" ? optionalString(contentRecord.text) : undefined;
      return text ? [text] : [];
    });
  });
  return parts.length > 0 ? parts.join("") : null;
}

function readUsage(value: unknown): NormalizedChatUsage {
  const usage = optionalRecord(value);
  return {
    promptTokens: optionalInteger(usage?.prompt_tokens) ?? null,
    completionTokens: optionalInteger(usage?.completion_tokens) ?? null,
    totalTokens: optionalInteger(usage?.total_tokens) ?? null,
  };
}

function readModelRecords(payload: Record<string, unknown>): Array<{ id: string; ownedBy: string | null }> {
  const items = Array.isArray(payload.data) ? payload.data : [];
  return items.flatMap((item) => {
    const record = optionalRecord(item);
    const id = optionalString(record?.id);
    if (!id) {
      return [];
    }
    return [{ id, ownedBy: optionalString(record?.owned_by) ?? null }];
  });
}

function readModelIds(payload: Record<string, unknown>): string[] {
  return readModelRecords(payload).map((model) => model.id);
}

/**
 * Read one relay answer under the shared JSON response cap. Image answers carry
 * either a URL or inline base64, so the cap bounds the body the connector can
 * be made to buffer; an oversized answer fails as 413 instead of growing.
 */
async function readOpenAiCompatibleResponseRecord(response: Response): Promise<Record<string, unknown>> {
  const parsed: unknown = await readProviderJson(response, imageApiLabel, defaultProviderJsonMaxResponseBytes);
  return requiredResponseRecord(parsed, `${imageApiLabel} response`);
}

async function requestOpenAiCompatibleJson(input: OpenAiCompatibleJsonRequest): Promise<Record<string, unknown>> {
  try {
    return await runProviderRequest(
      {
        signal: input.context.signal,
        label: imageApiLabel,
        timeoutMs: input.timeoutMs,
      },
      async (signal) => {
        const url = new URL(input.path, input.context.baseUrl).toString();
        const response = await sendOpenAiCompatibleRequest(input, url, input.body, signal);
        if (response.ok) {
          return readOpenAiCompatibleResponseRecord(response);
        }
        const text = await readProviderErrorTextBody(response, `${imageApiLabel} error response`);
        if (input.fallbackBody === undefined || !isFormRejection(response.status, text)) {
          throw new ProviderRequestError(response.status, text || `${imageApiLabel} request failed`);
        }
        const retry = await sendOpenAiCompatibleRequest(input, url, input.fallbackBody, signal);
        return readOpenAiCompatibleResponseRecord(retry);
      },
    );
  } catch (error) {
    // A validate-phase 401/403 becomes a field error on the connect form
    // instead of the reconnect prompt a 401/403 execution error produces.
    if (
      input.phase === "validate" &&
      error instanceof ProviderRequestError &&
      (error.status === 401 || error.status === 403)
    ) {
      throw new ProviderRequestError(400, error.message);
    }
    throw error;
  }
}

async function sendOpenAiCompatibleRequest(
  input: OpenAiCompatibleJsonRequest,
  url: string,
  body: Record<string, unknown> | FormData | undefined,
  signal: AbortSignal,
): Promise<Response> {
  const isForm = body instanceof FormData;
  return input.context.fetcher(url, {
    method: input.method,
    headers: compactObject({
      accept: "application/json",
      authorization: `Bearer ${input.context.apiKey}`,
      "content-type": body === undefined || isForm ? undefined : "application/json",
      "user-agent": providerUserAgent,
    }),
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
    signal,
  });
}

/**
 * A multipart edit request is rejected as a routing or parse failure by relays
 * that only implement the JSON shape. Authentication, rate-limit, and validation
 * answers are the relay's real decision and are never retried.
 */
function isFormRejection(status: number, text: string): boolean {
  if (status === 401 || status === 403 || status === 429) {
    return false;
  }
  if (status === 404 || status === 405 || status === 415) {
    return true;
  }
  const lower = text.toLowerCase();
  return lower.includes("multipart") || lower.includes("form data") || lower.includes("images api is not supported");
}
