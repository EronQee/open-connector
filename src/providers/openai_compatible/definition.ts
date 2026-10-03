import type { ProviderDefinition } from "../../core/types.ts";

import { openAiCompatibleActions } from "./actions.ts";

const service = "openai_compatible";

/**
 * OpenAI-compatible relay provider for API resale services that expose text,
 * image, and other models through the OpenAI `/v1` protocol under a
 * user-configured base URL.
 */
export const provider: ProviderDefinition = {
  service,
  displayName: "OpenAI-Compatible Relay",
  categories: ["AI", "Developer Tools"],
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
      ],
      testAction: {
        actionName: "list_models",
        input: {},
      },
    },
  ],
  actions: openAiCompatibleActions,
};
