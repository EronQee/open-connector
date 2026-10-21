import type { OpenAiCompatibleActionContext } from "./runtime.ts";

import { describe, expect, it } from "vitest";
import { ProviderRequestError } from "../provider-runtime.ts";
import { normalizeOpenAiCompatibleBaseUrl, openAiCompatibleActionHandlers } from "./runtime.ts";

interface RecordedCall {
  url: string;
  init: RequestInit;
}

function createContext(
  respond: (call: RecordedCall) => Response | Promise<Response>,
  onCall?: (call: RecordedCall) => void,
): OpenAiCompatibleActionContext {
  const fetcher = (async (input: string | URL | Request, init?: RequestInit) => {
    const call: RecordedCall = { url: String(input), init: init ?? {} };
    onCall?.(call);
    return respond(call);
  }) as typeof fetch;

  return {
    baseUrl: normalizeOpenAiCompatibleBaseUrl("https://api.example.com/v1"),
    apiKey: "test-key",
    fetcher,
  };
}

function json(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

describe("create_image", () => {
  it("collects images from the plain OpenAI envelope", async () => {
    const context = createContext(() =>
      json({ created: 1, data: [{ url: "https://cdn.example.com/a.png", revised_prompt: "a cat" }] }),
    );

    const result = (await openAiCompatibleActionHandlers.create_image(
      { model: "gpt-image-1", prompt: "a cat" },
      context,
    )) as { created: number; images: Array<{ url: string | null; revisedPrompt: string | null }> };

    expect(result.created).toBe(1);
    expect(result.images).toEqual([{ url: "https://cdn.example.com/a.png", b64Json: null, revisedPrompt: "a cat" }]);
  });

  it("collects images nested under data.result.images", async () => {
    const context = createContext(() =>
      json({ data: { result: { images: [{ url: "https://cdn.example.com/b.png" }] } } }),
    );

    const result = (await openAiCompatibleActionHandlers.create_image({ model: "m", prompt: "p" }, context)) as {
      images: Array<{ url: string | null }>;
    };

    expect(result.images.map((image) => image.url)).toEqual(["https://cdn.example.com/b.png"]);
  });

  it("collects Gemini inlineData parts", async () => {
    const context = createContext(() =>
      json({
        candidates: [{ content: { parts: [{ inlineData: { mimeType: "image/png", data: "QUJD" } }] } }],
      }),
    );

    const result = (await openAiCompatibleActionHandlers.create_image({ model: "m", prompt: "p" }, context)) as {
      images: Array<{ b64Json: string | null }>;
    };

    expect(result.images.map((image) => image.b64Json)).toEqual(["QUJD"]);
  });

  it("collects a data URL embedded in a text part", async () => {
    const context = createContext(() =>
      json({ candidates: [{ content: { parts: [{ text: "![img](data:image/png;base64,QUJD)" }] } }] }),
    );

    const result = (await openAiCompatibleActionHandlers.create_image({ model: "m", prompt: "p" }, context)) as {
      images: Array<{ b64Json: string | null }>;
    };

    expect(result.images.map((image) => image.b64Json)).toEqual(["QUJD"]);
  });

  it("merges extraBody into the request body", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ data: [{ url: "https://cdn.example.com/c.png" }] }),
      (call) => calls.push(call),
    );

    await openAiCompatibleActionHandlers.create_image(
      { model: "m", prompt: "p", extraBody: { resolution: "2k", official_fallback: false } },
      context,
    );

    expect(JSON.parse(String(calls[0]?.init.body))).toMatchObject({ resolution: "2k", official_fallback: false });
  });
});

describe("edit_image", () => {
  it("sends the reference images as multipart parts", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ data: [{ url: "https://cdn.example.com/edited.png" }] }),
      (call) => calls.push(call),
    );

    await openAiCompatibleActionHandlers.edit_image(
      { model: "m", prompt: "p", image: "data:image/png;base64,QUJD" },
      context,
    );

    const body = calls[0]?.init.body;
    expect(body).toBeInstanceOf(FormData);
    const form = body as FormData;
    expect(form.get("model")).toBe("m");
    expect(form.getAll("image[]")).toHaveLength(1);
    expect(calls[0]?.init.headers).not.toMatchObject({ "content-type": "application/json" });
  });

  it("retries as JSON when the relay rejects multipart routing", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      (call) => {
        if (call.init.body instanceof FormData) {
          return new Response(JSON.stringify({ error: { message: "unsupported media type" } }), { status: 415 });
        }
        return json({ data: [{ url: "https://cdn.example.com/edited.png" }] });
      },
      (call) => calls.push(call),
    );

    const result = (await openAiCompatibleActionHandlers.edit_image(
      { model: "m", prompt: "p", image: ["data:image/png;base64,QUJD"] },
      context,
    )) as { images: Array<{ url: string | null }> };

    expect(result.images.map((image) => image.url)).toEqual(["https://cdn.example.com/edited.png"]);
    expect(calls).toHaveLength(2);
    expect(JSON.parse(String(calls[1]?.init.body))).toMatchObject({ image: ["QUJD"] });
  });

  it("rejects a reference URL that is not publicly fetchable", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ data: [{ url: "https://cdn.example.com/edited.png" }] }),
      (call) => calls.push(call),
    );

    await expect(
      openAiCompatibleActionHandlers.edit_image({ model: "m", prompt: "p", image: "http://127.0.0.1/in.png" }, context),
    ).rejects.toMatchObject({ status: 400 });
    expect(calls).toHaveLength(0);
  });

  it("does not retry an authentication rejection", async () => {
    const context = createContext(() => new Response("unauthorized", { status: 401 }));

    await expect(
      openAiCompatibleActionHandlers.edit_image(
        { model: "m", prompt: "p", image: "data:image/png;base64,QUJD" },
        context,
      ),
    ).rejects.toBeInstanceOf(ProviderRequestError);
  });
});

describe("submit_image_task", () => {
  it("submits to the async route and normalizes a queued task", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ task_id: "task-1", status: "queued" }),
      (call) => calls.push(call),
    );

    const result = (await openAiCompatibleActionHandlers.submit_image_task({ model: "m", prompt: "p" }, context)) as {
      taskId: string;
      state: string;
      submitPath: string;
    };

    expect(calls[0]?.url).toBe("https://api.example.com/v1/images/generations/async");
    expect(result).toMatchObject({ taskId: "task-1", state: "queued", submitPath: "images/generations/async" });
  });

  it("honours a custom submit path", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ task_id: "task-1", status: "pending" }),
      (call) => calls.push(call),
    );

    await openAiCompatibleActionHandlers.submit_image_task(
      { model: "m", prompt: "p", submitPath: "images/generations" },
      context,
    );

    expect(calls[0]?.url).toBe("https://api.example.com/v1/images/generations");
  });
});

describe("get_image_task", () => {
  it("falls through 404 routes to the path the relay answers on", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      (call) => {
        if (call.url.includes("/v1/tasks/")) {
          return new Response("not found", { status: 404 });
        }
        return json({ status: "succeeded", data: [{ url: "https://cdn.example.com/done.png" }] });
      },
      (call) => calls.push(call),
    );

    const result = (await openAiCompatibleActionHandlers.get_image_task({ taskId: "task-1" }, context)) as {
      state: string;
      queryPath: string;
      images: Array<{ url: string | null }>;
    };

    expect(calls.map((call) => call.url)).toEqual([
      "https://api.example.com/v1/tasks/task-1",
      "https://api.example.com/v1/images/tasks/task-1",
    ]);
    expect(result).toMatchObject({ state: "completed", queryPath: "images/tasks/task-1" });
    expect(result.images.map((image) => image.url)).toEqual(["https://cdn.example.com/done.png"]);
  });

  it("treats a failure status delivered with HTTP 200 as failed", async () => {
    const context = createContext(() =>
      json({ status: "failed", error: { code: "content_policy", message: "blocked" } }),
    );

    const result = (await openAiCompatibleActionHandlers.get_image_task({ taskId: "t" }, context)) as {
      state: string;
      error: { code: string; message: string } | null;
    };

    expect(result.state).toBe("failed");
    expect(result.error).toEqual({ code: "content_policy", message: "blocked" });
  });

  it("treats images returned without a status as completed", async () => {
    const context = createContext(() => json({ data: [{ url: "https://cdn.example.com/e.png" }] }));

    const result = (await openAiCompatibleActionHandlers.get_image_task({ taskId: "t" }, context)) as {
      state: string;
    };

    expect(result.state).toBe("completed");
  });

  it("reports a non-404 failure instead of probing further", async () => {
    const context = createContext(() => new Response("rate limited", { status: 429 }));

    await expect(openAiCompatibleActionHandlers.get_image_task({ taskId: "t" }, context)).rejects.toMatchObject({
      status: 429,
    });
  });
});
