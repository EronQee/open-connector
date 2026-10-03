import type { ActionDefinition } from "../../core/types.ts";

import { s } from "../../core/json-schema.ts";
import { defineProviderAction } from "../../core/provider-definition.ts";

const service = "openai_compatible";

const chatMessageSchema = s.object("One chat message.", {
  role: s.string("The message role, such as system, user, or assistant."),
  content: s.unknown("The message content: a string, or the multimodal content parts the model accepts."),
});

const usageSchema = s.object(
  "The token usage reported by the relay, when returned.",
  {
    promptTokens: s.nullableInteger("The number of prompt tokens consumed."),
    completionTokens: s.nullableInteger("The number of completion tokens produced."),
    totalTokens: s.nullableInteger("The total number of tokens consumed."),
  },
  { optional: ["promptTokens", "completionTokens", "totalTokens"] },
);

const imageItemSchema = s.object(
  "One generated or edited image.",
  {
    url: s.nullableString("The hosted image URL when the relay returned one."),
    b64Json: s.nullableString("The base64-encoded image data when the relay returned image bytes."),
    revisedPrompt: s.nullableString("The prompt as rewritten by the upstream model, when returned."),
  },
  { optional: ["url", "b64Json", "revisedPrompt"] },
);

const imageResponseSchema = s.object("The normalized image generation output.", {
  created: s.nullableInteger("The Unix timestamp when the relay created the images."),
  images: s.array("The images returned by the relay.", imageItemSchema),
  raw: s.looseObject("The raw image API response payload."),
});

const imageRequestFields = {
  n: s.integer("The number of images to generate.", { minimum: 1 }),
  size: s.string("The requested image size as widthxheight, such as 1024x1024. Supported values depend on the model."),
  quality: s.string(
    "The image quality to request. Relays accept values such as low, medium, high, or auto depending on the model.",
  ),
  style: s.string("The image style to request, passed through to models that support it."),
  background: s.string("The background treatment to request, such as transparent, opaque, or auto."),
  outputFormat: s.stringEnum(["png", "jpeg", "webp"], { description: "The image file format to request." }),
  responseFormat: s.stringEnum(["url", "b64_json"], {
    description: "The image payload format to request. Some models only support b64_json.",
  }),
  watermark: s.boolean(
    "Whether to apply a watermark. Relays that support the field treat an explicit false differently from an omitted value.",
  ),
};

const optionalImageRequestFields = [
  "n",
  "size",
  "quality",
  "style",
  "background",
  "outputFormat",
  "responseFormat",
  "watermark",
];

const listModelsAction = defineProviderAction(service, {
  name: "list_models",
  operationType: "read",
  description:
    "List the models the relay API key can access. Use the returned model ids verbatim in generation actions.",
  inputSchema: s.object("The input payload for listing relay models.", {}),
  outputSchema: s.object("The normalized model list output.", {
    models: s.array(
      "The models available to this API key.",
      s.object("One available model.", {
        id: s.string("The model identifier to send as the model request field."),
        ownedBy: s.nullableString("The owner of the model when the relay reports one."),
      }),
    ),
    total: s.integer("The number of models returned."),
    raw: s.looseObject("The raw models response payload."),
  }),
});

const createChatCompletionAction = defineProviderAction(service, {
  name: "create_chat_completion",
  operationType: "write",
  description:
    "Create a chat completion with a text model exposed by the relay using the OpenAI chat completions protocol.",
  inputSchema: s.object(
    "The input payload for creating a chat completion.",
    {
      model: s.string("The text model to use, as listed by the relay."),
      messages: s.array("The conversation so far.", chatMessageSchema, { minItems: 1 }),
      temperature: s.number("The sampling temperature to request."),
      topP: s.number("The nucleus sampling probability mass to request."),
      maxTokens: s.integer("The maximum number of tokens to generate."),
      stop: s.union([
        s.string("One stop sequence."),
        s.array("Up to four stop sequences.", s.string("One stop sequence.")),
      ]),
    },
    { optional: ["temperature", "topP", "maxTokens", "stop"] },
  ),
  outputSchema: s.object("The normalized chat completion output.", {
    id: s.nullableString("The completion identifier when the relay returns one."),
    model: s.nullableString("The model that produced the completion."),
    message: s.object("The assistant message that answers the conversation.", {
      role: s.nullableString("The message role, usually assistant."),
      content: s.unknown("The message content: a string, or the content parts the model returned."),
    }),
    finishReason: s.nullableString("Why generation stopped, such as stop or length."),
    usage: usageSchema,
    raw: s.looseObject("The raw chat completion response payload."),
  }),
});

const createResponseAction = defineProviderAction(service, {
  name: "create_response",
  operationType: "write",
  description:
    "Create a model response using the OpenAI Responses protocol. Use this for relay models served through /v1/responses.",
  inputSchema: s.object(
    "The input payload for creating a Responses API generation.",
    {
      model: s.string("The model to use, as listed by the relay."),
      input: s.union(
        [
          s.string("The user input text."),
          s.array("The structured input items the Responses API accepts.", s.looseObject("One Responses input item.")),
        ],
        { description: "The input to respond to: a string or structured input items." },
      ),
      instructions: s.string("Top-level instructions that apply across the conversation."),
      maxOutputTokens: s.integer("The maximum number of output tokens to generate."),
      previousResponseId: s.string("The identifier of the previous response to continue from."),
      store: s.boolean("Whether the relay should store the response for later retrieval."),
    },
    { optional: ["instructions", "maxOutputTokens", "previousResponseId", "store"] },
  ),
  outputSchema: s.object("The normalized Responses API output.", {
    id: s.nullableString("The response identifier."),
    status: s.nullableString("The response status, such as completed."),
    outputText: s.nullableString("The assistant text joined from the response output messages."),
    usage: usageSchema,
    raw: s.looseObject("The raw Responses API response payload."),
  }),
});

const createImageAction = defineProviderAction(service, {
  name: "create_image",
  operationType: "write",
  description:
    "Generate images with a relay image model using the OpenAI images generations protocol. The response is synchronous.",
  inputSchema: s.object(
    "The input payload for generating images.",
    {
      model: s.string("The image model to use, such as gpt-image-1 or a relay-specific model id."),
      prompt: s.string("The prompt describing the image to generate."),
      ...imageRequestFields,
    },
    { optional: optionalImageRequestFields },
  ),
  outputSchema: imageResponseSchema,
});

const editImageAction = defineProviderAction(service, {
  name: "edit_image",
  operationType: "write",
  description:
    "Edit existing images with a relay image model using the OpenAI images edits protocol. Reference images are sent as base64 data or public HTTPS URLs in the JSON body.",
  inputSchema: s.object(
    "The input payload for editing images.",
    {
      model: s.string("The image model to use."),
      prompt: s.string("The instruction describing the edit to apply."),
      image: s.union(
        [
          s.string("One reference image as a base64 data string or public HTTPS URL."),
          s.array(
            "Multiple reference images.",
            s.string("One reference image as a base64 data string or public HTTPS URL."),
          ),
        ],
        { description: "The reference image or images to edit." },
      ),
      mask: s.string("An optional mask image as base64 data or a public HTTPS URL marking the area to edit."),
      ...imageRequestFields,
    },
    { optional: ["mask", ...optionalImageRequestFields] },
  ),
  outputSchema: imageResponseSchema,
});

export const openAiCompatibleActions: ActionDefinition[] = [
  listModelsAction,
  createChatCompletionAction,
  createResponseAction,
  createImageAction,
  editImageAction,
];
