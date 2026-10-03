import type { ProviderDefinition } from "../../core/types.ts";

import { openAiVideoActions } from "./actions.ts";

const service = "openai_video";

/**
 * OpenAI-compatible relay provider for asynchronous video generation task
 * APIs (OpenAI/Sora-style /videos and Seedance/Doubao-style
 * /video/generations) under a user-configured base URL.
 */
export const provider: ProviderDefinition = {
  service,
  displayName: "OpenAI-Compatible Video Relay",
  categories: ["AI"],
  authTypes: ["custom_credential"],
  auth: [
    {
      type: "custom_credential",
      fields: [
        {
          key: "baseUrl",
          label: "API Base URL",
          inputType: "text",
          required: true,
          secret: false,
          placeholder: "https://api.example.com/v1",
          description:
            "The relay base URL, such as https://api.example.com/v1 or https://relay.example.cn. A bare host gets /v1 appended automatically.",
        },
        {
          key: "apiKey",
          label: "API Key",
          inputType: "password",
          required: true,
          secret: true,
          placeholder: "sk-...",
          description: "The relay API key sent with the Authorization Bearer header.",
        },
        {
          key: "protocol",
          label: "Task Protocol",
          inputType: "text",
          required: false,
          secret: false,
          placeholder: "openai",
          description:
            "The task protocol the relay uses for videos: openai (POST /v1/videos, used by Sora-protocol and MiniMax-style relays) or seedance (POST /v1/video/generations, used by Doubao Seedance relays). Defaults to openai.",
        },
      ],
      testAction: {
        actionName: "get_video",
        input: {
          taskId: "connectivity-probe",
        },
      },
    },
  ],
  actions: openAiVideoActions,
};
