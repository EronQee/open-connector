import type { CredentialValidationResult, TransitFileUpload, TransitFileWriter } from "../../core/types.ts";
import type { ProviderActionHandlers } from "../provider-runtime.ts";

import { compactObject, optionalNumber, optionalRecord, optionalString } from "../../core/cast.ts";
import { assertPublicHttpUrl, encodePathSegment, readBoundedResponseBytes } from "../../core/request.ts";
import { normalizeOpenAiCompatibleBaseUrl } from "../openai_compatible/runtime.ts";
import {
  ProviderRequestError,
  providerFetch,
  providerUserAgent,
  readProviderErrorTextBody,
  readProviderJson,
  requiredInputString,
  requiredResponseRecord,
  runProviderRequest,
} from "../provider-runtime.ts";

/**
 * Video downloads can run for minutes; the task submission and polling
 * endpoints stay on the default 30 second provider budget.
 */
const videoDownloadTimeoutMs = 600_000;

const openaiProtocolTaskPath = "videos";
const seedanceProtocolTaskPath = "video/generations";

/**
 * Relays that share a submit route still disagree on where the task record
 * lives. A poll is idempotent and a routing 404 says nothing about the task,
 * so the query tries the protocol path first and then the general task routes
 * before reporting failure.
 */
const openaiProtocolTaskQueryPaths = ["videos", "video/generations", "tasks", "videos/generations"];
const seedanceProtocolTaskQueryPaths = ["video/generations", "videos", "tasks", "videos/generations"];

const videoStatusInProgress = new Set([
  "pending",
  "running",
  "processing",
  "in_progress",
  "in-progress",
  "generating",
  "submitted",
  "created",
]);
const videoStatusCompleted = new Set([
  "succeeded",
  "success",
  "successful",
  "completed",
  "complete",
  "finish",
  "finished",
  "done",
  "ok",
  "ready",
]);
const videoStatusFailed = new Set([
  "failed",
  "failure",
  "fail",
  "error",
  "errored",
  "cancelled",
  "canceled",
  "timeout",
  "timedout",
  "rejected",
  "expired",
  "deleted",
]);

export type OpenAiVideoProtocol = "openai" | "seedance";
export type NormalizedVideoState = "queued" | "in_progress" | "completed" | "failed" | "unknown";

type OpenAiVideoActionHandler = (input: Record<string, unknown>, context: OpenAiVideoActionContext) => Promise<unknown>;

export interface OpenAiVideoActionContext {
  baseUrl: string;
  apiKey: string;
  protocol: OpenAiVideoProtocol;
  fetcher: typeof fetch;
  signal?: AbortSignal;
  transitFiles?: TransitFileWriter;
}

interface NormalizedVideoError {
  code: string | null;
  message: string | null;
}

interface NormalizedVideoTask {
  taskId: string | null;
  status: string | null;
  state: NormalizedVideoState;
  progress: number | null;
  videoUrl: string | null;
  error: NormalizedVideoError | null;
}

export const openAiVideoActionHandlers: ProviderActionHandlers<"openai_video", OpenAiVideoActionHandler> = {
  create_video(input, context) {
    return createVideo(input, context);
  },
  get_video(input, context) {
    return getVideo(input, context);
  },
};

export async function validateOpenAiVideoCredential(
  input: Record<string, string>,
  fetcher: typeof fetch,
  signal?: AbortSignal,
): Promise<CredentialValidationResult> {
  const context = createOpenAiVideoContext(input, fetcher, signal);
  let probeStatus: number | null = null;
  try {
    await requestOpenAiVideoJson({
      context,
      method: "GET",
      path: `${readTaskPath(context.protocol)}/connectivity-probe`,
    });
  } catch (error) {
    // Video relays answer an unknown task id with anything from 200-plus
    // error JSON to 404, so any upstream answer that is not an auth rejection
    // proves the base URL and protocol path are reachable with this key.
    if (error instanceof ProviderRequestError && (error.status === 401 || error.status === 403)) {
      throw new ProviderRequestError(400, error.message);
    }
    if (error instanceof ProviderRequestError && error.status >= 400 && error.status < 500) {
      probeStatus = error.status;
    } else {
      throw error;
    }
  }

  return {
    profile: {
      accountId: new URL(context.baseUrl).host,
      displayName: `OpenAI-compatible video API @ ${new URL(context.baseUrl).host} (${context.protocol} protocol)`,
    },
    grantedScopes: [],
    metadata: compactObject({
      baseUrl: context.baseUrl,
      protocol: context.protocol,
      probeStatus,
    }),
  };
}

export function createOpenAiVideoContext(
  input: Record<string, string>,
  fetcher: typeof fetch,
  signal?: AbortSignal,
  transitFiles?: TransitFileWriter,
): OpenAiVideoActionContext {
  return {
    baseUrl: normalizeOpenAiCompatibleBaseUrl(input.baseUrl),
    apiKey: requiredInputString(input.apiKey, "apiKey"),
    protocol: readProtocol(input.protocol),
    fetcher,
    signal,
    transitFiles,
  };
}

async function createVideo(input: Record<string, unknown>, context: OpenAiVideoActionContext): Promise<unknown> {
  const body = buildCreateVideoBody(input, context.protocol);
  const payload = await requestOpenAiVideoJson({
    context,
    method: "POST",
    path: readTaskPath(context.protocol),
    body,
  });

  return normalizeVideoTask(payload, "video task submission response");
}

async function getVideo(input: Record<string, unknown>, context: OpenAiVideoActionContext): Promise<unknown> {
  const taskId = requiredInputString(input.taskId, "taskId");
  const { payload, queryPath } = await readVideoTaskPayload(taskId, input.queryPath, context);
  const task = normalizeVideoTask(payload, "video task response");
  const file = input.download === true ? await downloadVideoToTransitFile(task, context) : null;

  return {
    ...task,
    taskId: task.taskId ?? taskId,
    queryPath,
    file,
    raw: payload,
  };
}

/**
 * Query the task record across the paths relays put it under. A 404 means that
 * path is not this relay's route, so the next candidate is tried; every other
 * failure is the relay answering about the task and is reported as-is.
 */
async function readVideoTaskPayload(
  taskId: string,
  rawQueryPath: unknown,
  context: OpenAiVideoActionContext,
): Promise<{ payload: Record<string, unknown>; queryPath: string }> {
  const overridden = optionalString(rawQueryPath);
  const candidates = overridden ? [overridden] : readTaskQueryPaths(context.protocol);
  let lastNotFound: ProviderRequestError | undefined;

  for (const candidate of candidates) {
    const queryPath = `${candidate.replace(/\/+$/, "")}/${encodePathSegment(taskId)}`;
    try {
      const payload = await requestOpenAiVideoJson({
        context,
        method: "GET",
        path: queryPath,
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

  throw lastNotFound ?? new ProviderRequestError(502, "No video task query path answered for this relay.");
}

/**
 * Build the video task body for the connection's protocol. Relays share the
 * asynchronous task lifecycle but express prompt, duration, and reference
 * media differently, so common fields map per protocol and `extraBody`
 * merges last with documented override semantics.
 */
function buildCreateVideoBody(input: Record<string, unknown>, protocol: OpenAiVideoProtocol): Record<string, unknown> {
  const seconds = optionalNumber(input.seconds);
  const images = readReferenceImageUrls(input.images);
  const body: Record<string, unknown> = compactObject({
    model: requiredInputString(input.model, "model"),
    prompt: requiredInputString(input.prompt, "prompt"),
    seconds: seconds === undefined ? undefined : protocol === "seedance" ? String(seconds) : seconds,
  });
  if (protocol === "seedance") {
    const metadata = compactObject({
      content: images?.map((url) => ({ type: "image_url", image_url: { url }, role: "reference_image" })),
      resolution: optionalString(input.resolution),
      ratio: optionalString(input.aspectRatio),
    });
    if (Object.keys(metadata).length > 0) {
      body.metadata = metadata;
    }
  } else {
    if (images && images.length > 0) {
      body.images = images;
    }
    if (optionalString(input.resolution)) {
      body.resolution = input.resolution;
    }
    if (optionalString(input.aspectRatio)) {
      body.aspect_ratio = input.aspectRatio;
    }
  }
  const extraBody = optionalRecord(input.extraBody);
  if (extraBody) {
    Object.assign(body, extraBody);
  }
  return body;
}

function normalizeVideoTask(payload: Record<string, unknown>, source: string): NormalizedVideoTask {
  const task =
    optionalRecord(payload.task) ??
    optionalRecord(payload.data) ??
    optionalRecord(optionalRecord(payload.data)?.task) ??
    optionalRecord(payload.detail) ??
    payload;
  const status =
    optionalString(task.status) ?? optionalString(task.task_status) ?? optionalString(payload.status) ?? null;
  const videoUrl = readVideoUrl(payload, task);
  const normalized: NormalizedVideoTask = {
    taskId:
      optionalString(task.id) ??
      optionalString(task.task_id) ??
      optionalString(task.taskId) ??
      optionalString(payload.id) ??
      optionalString(payload.task_id) ??
      null,
    status,
    // A relay that already handed back a playable URL has finished even when
    // its status field is missing or non-standard.
    state: normalizeVideoStatus(status ?? undefined, videoUrl !== null),
    progress: optionalNumber(task.progress) ?? optionalNumber(payload.progress) ?? null,
    videoUrl,
    error: readVideoError(task.error) ?? readVideoError(payload.error),
  };
  if (!normalized.taskId && normalized.state === "unknown") {
    throw new ProviderRequestError(502, `Could not read a task id or status from the ${source}`);
  }
  return normalized;
}

/**
 * Collect the download URL from the places relay protocols disagree on:
 * lens returns a top-level `download_url`, MiniMax-style relays nest it in
 * `task.content.url`, and Seedance-style relays put `result_url` on the data
 * record.
 */
function readVideoUrl(payload: Record<string, unknown>, task: Record<string, unknown>): string | null {
  const data = optionalRecord(payload.data);
  const candidates = [
    optionalString(payload.download_url),
    optionalString(task.download_url),
    optionalString(task.content_url),
    optionalString(optionalRecord(task.content)?.url),
    optionalString(optionalRecord(payload.content)?.url),
    optionalString(payload.video_url),
    optionalString(task.video_url),
    optionalString(payload.url),
    optionalString(task.url),
    optionalString(task.result_url),
    optionalString(data?.result_url),
    optionalString(data?.video_url),
    optionalString(data?.url),
    optionalString(optionalRecord(data?.content)?.url),
    optionalString(optionalRecord(optionalRecord(data?.result)?.content)?.url),
    firstVideoUrl(optionalRecord(task.result)),
    firstVideoUrl(optionalRecord(data?.result)),
  ];
  return candidates.find((candidate): candidate is string => Boolean(candidate)) ?? null;
}

/**
 * Some relays answer with `result.videos[]` or `result.outputs[]` instead of a
 * single URL field.
 */
function firstVideoUrl(result: Record<string, unknown> | undefined): string | undefined {
  if (!result) {
    return undefined;
  }
  for (const key of ["videos", "outputs", "results", "content"]) {
    const value = result[key];
    const first = Array.isArray(value) ? value.at(0) : value;
    const direct = optionalString(first);
    if (direct) {
      return direct;
    }
    const record = optionalRecord(first);
    const nested =
      optionalString(record?.url) ??
      optionalString(record?.video_url) ??
      optionalString(optionalRecord(record?.content)?.url);
    if (nested) {
      return nested;
    }
  }
  return undefined;
}

function readVideoError(value: unknown): NormalizedVideoError | null {
  const record = optionalRecord(value);
  if (!record) {
    return null;
  }
  const message = optionalString(record.message);
  const code = optionalString(record.code);
  if (!message && !code) {
    return null;
  }
  return { code: code ?? null, message: message ?? null };
}

function normalizeVideoStatus(status: string | undefined, hasVideoUrl = false): NormalizedVideoState {
  const key = status?.trim().toLowerCase() ?? "";
  if (key && videoStatusFailed.has(key)) {
    return "failed";
  }
  if (key && videoStatusCompleted.has(key)) {
    return "completed";
  }
  if (!key) {
    return hasVideoUrl ? "completed" : "unknown";
  }
  if (key === "queued") {
    return "queued";
  }
  if (videoStatusInProgress.has(key)) {
    return "in_progress";
  }
  return "unknown";
}

/**
 * Download the completed video into transit file storage. The bytes stay out
 * of action output otherwise: the default flow hands back the relay's URL and
 * the caller fetches the file to its own storage. A public signed URL is
 * fetched without credentials; a content endpoint on the relay origin that
 * requires the API key is retried with the Bearer header.
 */
async function downloadVideoToTransitFile(
  task: NormalizedVideoTask,
  context: OpenAiVideoActionContext,
): Promise<TransitFileUpload | null> {
  if (!context.transitFiles) {
    throw new ProviderRequestError(400, "Transit file storage is not enabled for this deployment.");
  }
  if (task.state !== "completed") {
    throw new ProviderRequestError(400, "The video task has not completed; poll get_video before downloading.");
  }
  if (!task.videoUrl) {
    throw new ProviderRequestError(502, "The video task response did not include a downloadable video URL.");
  }

  const response = await runProviderRequest(
    { signal: context.signal, label: "OpenAI-compatible video download", timeoutMs: videoDownloadTimeoutMs },
    async (signal) => {
      let download = await providerFetch(task.videoUrl!, {
        headers: { accept: "*/*", "user-agent": providerUserAgent },
        signal,
      });
      if (!download.ok && isSameOrigin(task.videoUrl!, context.baseUrl)) {
        download = await context.fetcher(task.videoUrl!, {
          headers: { accept: "*/*", authorization: `Bearer ${context.apiKey}`, "user-agent": providerUserAgent },
          signal,
        });
      }
      return download;
    },
  );
  if (!response.ok) {
    const text = await readProviderErrorTextBody(response, "video download error response");
    throw new ProviderRequestError(response.status >= 500 ? 502 : response.status, text || "video download failed");
  }
  const bytes = await readBoundedResponseBytes(response, {
    maxBytes: context.transitFiles.maxBytes,
    fieldName: "video",
    createError: (message) => new ProviderRequestError(413, message),
  });
  const mimeType = response.headers.get("content-type") ?? "video/mp4";
  const name = `${task.taskId ?? "video"}.mp4`;
  const upload = await context.transitFiles.create(new File([Uint8Array.from(bytes)], name, { type: mimeType }));

  return {
    fileId: upload.fileId,
    downloadUrl: upload.downloadUrl,
    sizeBytes: upload.sizeBytes,
    name,
    mimeType,
  };
}

function isSameOrigin(url: string, baseUrl: string): boolean {
  try {
    return new URL(url).origin === new URL(baseUrl).origin;
  } catch {
    return false;
  }
}

function readReferenceImageUrls(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const urls = value.flatMap((item) => {
    const url = optionalString(item);
    if (!url) {
      return [];
    }
    // Reference media are fetched by the relay during generation, and every
    // relay documents public HTTPS URLs only.
    assertPublicHttpUrl(url, {
      fieldName: "images",
      createError: (message) => new ProviderRequestError(400, message),
    });
    return [url];
  });
  return urls.length > 0 ? urls : undefined;
}

function readProtocol(value: unknown): OpenAiVideoProtocol {
  const protocol = optionalString(value) ?? "openai";
  if (protocol !== "openai" && protocol !== "seedance") {
    throw new ProviderRequestError(400, "protocol must be openai or seedance");
  }
  return protocol;
}

function readTaskPath(protocol: OpenAiVideoProtocol): string {
  return protocol === "seedance" ? seedanceProtocolTaskPath : openaiProtocolTaskPath;
}

function readTaskQueryPaths(protocol: OpenAiVideoProtocol): string[] {
  return protocol === "seedance" ? seedanceProtocolTaskQueryPaths : openaiProtocolTaskQueryPaths;
}

async function requestOpenAiVideoJson(input: {
  context: OpenAiVideoActionContext;
  method: "GET" | "POST";
  path: string;
  body?: Record<string, unknown>;
}): Promise<Record<string, unknown>> {
  return runProviderRequest({ signal: input.context.signal, label: "OpenAI-compatible video API" }, async (signal) => {
    const response = await input.context.fetcher(new URL(input.path, input.context.baseUrl).toString(), {
      method: input.method,
      headers: compactObject({
        accept: "application/json",
        authorization: `Bearer ${input.context.apiKey}`,
        "content-type": input.body === undefined ? undefined : "application/json",
        "user-agent": providerUserAgent,
      }),
      body: input.body === undefined ? undefined : JSON.stringify(input.body),
      signal,
    });
    const parsed: unknown = await readProviderJson(response, "OpenAI-compatible video API");
    return requiredResponseRecord(parsed, "OpenAI-compatible video API response");
  });
}
