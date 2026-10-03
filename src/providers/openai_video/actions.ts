import type { ActionDefinition } from "../../core/types.ts";

import { s } from "../../core/json-schema.ts";
import { defineProviderAction } from "../../core/provider-definition.ts";

const service = "openai_video";

const videoErrorSchema = s.object(
  "The error the relay reported for the task.",
  {
    code: s.nullableString("The upstream error code, when returned."),
    message: s.nullableString("The upstream error message, when returned."),
  },
  { optional: ["code", "message"] },
);

const videoFileSchema = s.looseObject(
  "The video stored in the connector's transit file storage when download was requested.",
);

const videoTaskFields = {
  status: s.nullableString("The task status verbatim as the relay returned it."),
  state: s.stringEnum(["queued", "in_progress", "completed", "failed", "unknown"], {
    description:
      "The task status normalized across relay protocols. Relays answer failures with HTTP 200, so poll this field rather than the HTTP status.",
  }),
  progress: s.nullableNumber(
    "The task progress as reported by the relay: a fraction or a percentage depending on the relay.",
  ),
  videoUrl: s.nullableString(
    "The download URL for the completed video when the relay returned one. Fetch this URL from your own environment; the connector does not transfer the file.",
  ),
  error: s.nullable(videoErrorSchema),
};

const optionalVideoTaskFields = ["status", "state", "progress", "videoUrl", "error"];

const createVideoAction = defineProviderAction(service, {
  name: "create_video",
  operationType: "write",
  description:
    "Submit an asynchronous video generation task to a relay. Poll get_video with the returned taskId until state is completed, then download from videoUrl.",
  inputSchema: s.object(
    "The input payload for submitting a video generation task.",
    {
      model: s.string("The video model to use, as listed by the relay."),
      prompt: s.string("The prompt describing the video, including camera and audio direction when supported."),
      seconds: s.number("The requested video duration in seconds."),
      resolution: s.string(
        "The requested output resolution, such as 720p or 768P. Must match what the model supports.",
      ),
      aspectRatio: s.string("The requested aspect ratio, such as 16:9 or 9:16."),
      images: s.array(
        "Reference image URLs the relay downloads during generation. Must be public HTTPS URLs.",
        s.string("One public HTTPS reference image URL."),
        { minItems: 1 },
      ),
      extraBody: s.record(
        "Protocol-specific fields merged into the request body verbatim, overriding the generated ones. Use this for relay-specific fields such as MiniMax-style content[] arrays or Seedance watermark and generate_audio settings.",
        s.unknown("Any JSON value the relay accepts."),
      ),
    },
    { optional: ["model", "prompt", "seconds", "resolution", "aspectRatio", "images", "extraBody"] },
  ),
  outputSchema: s.object(
    "The normalized task submission output.",
    {
      taskId: s.nullableString("The task identifier to poll with get_video."),
      ...videoTaskFields,
      raw: s.looseObject("The raw task submission response payload."),
    },
    { optional: optionalVideoTaskFields },
  ),
});

const getVideoAction = defineProviderAction(service, {
  name: "get_video",
  operationType: "read",
  description:
    "Query the status of a relay video generation task. Relays report failures with HTTP 200, so branch on state, not on the call succeeding.",
  inputSchema: s.object(
    "The input payload for querying a video generation task.",
    {
      taskId: s.string("The task identifier returned by create_video."),
      download: s.boolean(
        "Whether to transfer the completed video into the connector's transit file storage. Leave off to fetch videoUrl from your own environment; only turn this on when the relay's download URL requires the API key or expires too quickly.",
      ),
    },
    { optional: ["taskId", "download"] },
  ),
  outputSchema: s.object(
    "The normalized task status output.",
    {
      taskId: s.string("The task identifier polled."),
      ...videoTaskFields,
      file: s.nullable(videoFileSchema),
      raw: s.looseObject("The raw task status response payload."),
    },
    { optional: optionalVideoTaskFields },
  ),
});

export const openAiVideoActions: ActionDefinition[] = [createVideoAction, getVideoAction];
