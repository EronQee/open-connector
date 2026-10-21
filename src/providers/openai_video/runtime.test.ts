import type { OpenAiVideoActionContext } from "./runtime.ts";

import { describe, expect, it } from "vitest";
import { normalizeOpenAiCompatibleBaseUrl } from "../openai_compatible/runtime.ts";
import { openAiVideoActionHandlers } from "./runtime.ts";

interface RecordedCall {
  url: string;
  init: RequestInit;
}

function createContext(
  respond: (call: RecordedCall) => Response | Promise<Response>,
  onCall?: (call: RecordedCall) => void,
): OpenAiVideoActionContext {
  const fetcher = (async (input: string | URL | Request, init?: RequestInit) => {
    const call: RecordedCall = { url: String(input), init: init ?? {} };
    onCall?.(call);
    return respond(call);
  }) as typeof fetch;

  return {
    baseUrl: normalizeOpenAiCompatibleBaseUrl("https://relay.example.com/v1"),
    apiKey: "test-key",
    protocol: "openai",
    fetcher,
  };
}

function json(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

describe("get_video", () => {
  it("falls through 404 routes to the path the relay answers on", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      (call) => {
        if (call.url.includes("/v1/videos/")) {
          return new Response("not found", { status: 404 });
        }
        return json({ status: "completed", video_url: "https://cdn.example.com/out.mp4" });
      },
      (call) => calls.push(call),
    );

    const result = (await openAiVideoActionHandlers.get_video({ taskId: "t1" }, context)) as {
      state: string;
      queryPath: string;
      videoUrl: string | null;
    };

    expect(calls[0]?.url).toBe("https://relay.example.com/v1/videos/t1");
    expect(result).toMatchObject({ state: "completed", queryPath: "video/generations/t1" });
    expect(result.videoUrl).toBe("https://cdn.example.com/out.mp4");
  });

  it("honours an explicit query path instead of probing", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ status: "running" }),
      (call) => calls.push(call),
    );

    await openAiVideoActionHandlers.get_video({ taskId: "t1", queryPath: "tasks" }, context);

    expect(calls.map((call) => call.url)).toEqual(["https://relay.example.com/v1/tasks/t1"]);
  });

  it("marks a task completed when the relay returned a URL without a status", async () => {
    const context = createContext(() =>
      json({ data: { result: { videos: [{ url: "https://cdn.example.com/v.mp4" }] } } }),
    );

    const result = (await openAiVideoActionHandlers.get_video({ taskId: "t1" }, context)) as {
      state: string;
      videoUrl: string | null;
    };

    expect(result).toMatchObject({ state: "completed", videoUrl: "https://cdn.example.com/v.mp4" });
  });

  it("keeps a failed state even when the payload also carries a URL", async () => {
    const context = createContext(() => json({ status: "failed", video_url: "https://cdn.example.com/partial.mp4" }));

    const result = (await openAiVideoActionHandlers.get_video({ taskId: "t1" }, context)) as { state: string };

    expect(result.state).toBe("failed");
  });
});

describe("create_video", () => {
  it("submits to the openai protocol path by default", async () => {
    const calls: RecordedCall[] = [];
    const context = createContext(
      () => json({ id: "t1", status: "queued" }),
      (call) => calls.push(call),
    );

    const result = (await openAiVideoActionHandlers.create_video({ model: "sora-2", prompt: "waves" }, context)) as {
      taskId: string | null;
      state: string;
    };

    expect(calls[0]?.url).toBe("https://relay.example.com/v1/videos");
    expect(result).toMatchObject({ taskId: "t1", state: "queued" });
  });

  it("submits to the seedance path when the connection selects that protocol", async () => {
    const calls: RecordedCall[] = [];
    const context: OpenAiVideoActionContext = {
      ...createContext(
        () => json({ id: "t1", status: "queued" }),
        (call) => calls.push(call),
      ),
      protocol: "seedance",
    };

    await openAiVideoActionHandlers.create_video({ model: "doubao-seedance", prompt: "waves" }, context);

    expect(calls[0]?.url).toBe("https://relay.example.com/v1/video/generations");
  });
});
