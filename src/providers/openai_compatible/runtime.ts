import type { CredentialValidationResult } from "../../core/types.ts";
import type { ProviderActionHandlers } from "../provider-runtime.ts";

import {
  compactObject,
  optionalInteger,
  optionalNumber,
  optionalRecord,
  optionalString,
  optionalStringArray,
} from "../../core/cast.ts";
import { assertPublicHttpUrl, isPrivateNetworkAccessAllowed } from "../../core/request.ts";
import {
  ProviderRequestError,
  providerUserAgent,
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
  body?: Record<string, unknown>;
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
  const body = buildImageRequestBody(input, {
    model: requiredInputString(input.model, "model"),
    prompt: requiredInputString(input.prompt, "prompt"),
  });
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

async function editImage(input: Record<string, unknown>, context: OpenAiCompatibleActionContext) {
  const body = buildImageRequestBody(input, {
    model: requiredInputString(input.model, "model"),
    prompt: requiredInputString(input.prompt, "prompt"),
    image: optionalString(input.image) ?? optionalStringArray(input.image),
  });
  if (body.image === undefined) {
    throw new ProviderRequestError(400, "image is required");
  }
  if (optionalString(input.mask)) {
    body.mask = input.mask;
  }
  const payload = await requestOpenAiCompatibleJson({
    context,
    method: "POST",
    path: "images/edits",
    body,
    timeoutMs: imageGenerationTimeoutMs,
    phase: "execute",
  });

  return normalizeImageResponse(payload);
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
  const items = Array.isArray(payload.data) ? payload.data : [];
  const images = items.flatMap((item) => {
    const record = optionalRecord(item);
    if (!record) {
      return [];
    }
    const normalized: NormalizedImageItem = {
      url: optionalString(record.url) ?? null,
      b64Json: optionalString(record.b64_json) ?? null,
      revisedPrompt: optionalString(record.revised_prompt) ?? null,
    };
    return [normalized];
  });

  return {
    created: optionalInteger(payload.created) ?? null,
    images,
    raw: payload,
  };
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

async function requestOpenAiCompatibleJson(input: OpenAiCompatibleJsonRequest): Promise<Record<string, unknown>> {
  try {
    return await runProviderRequest(
      {
        signal: input.context.signal,
        label: "OpenAI-compatible API",
        timeoutMs: input.timeoutMs,
      },
      async (signal) => {
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
        const parsed: unknown = await readProviderJson(response, "OpenAI-compatible API");
        return requiredResponseRecord(parsed, "OpenAI-compatible API response");
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
