import type {
  CredentialValidationResult,
  CredentialValidators,
  ExecutionContext,
  ProviderExecutors,
} from "../../core/types.ts";
import type { OpenAiVideoActionContext } from "./runtime.ts";

import { isPrivateNetworkAccessAllowed } from "../../core/request.ts";
import { createProviderFetch, defineProviderExecutors, requireCustomCredential } from "../provider-runtime.ts";
import { createOpenAiVideoContext, openAiVideoActionHandlers, validateOpenAiVideoCredential } from "./runtime.ts";

const service = "openai_video";

export const executors: ProviderExecutors = defineProviderExecutors<OpenAiVideoActionContext>({
  service,
  handlers: openAiVideoActionHandlers,
  allowPrivateNetwork: isPrivateNetworkAccessAllowed,
  async createContext(context: ExecutionContext, fetcher: typeof fetch): Promise<OpenAiVideoActionContext> {
    const credential = await requireCustomCredential(context, service);
    return createOpenAiVideoContext(credential.values, fetcher, context.signal, context.transitFiles);
  },
  fallbackMessage: "unknown openai_video action",
});

export const credentialValidators: CredentialValidators = {
  customCredential(input, { fetcher, signal }): Promise<CredentialValidationResult> {
    const guardedFetcher = createProviderFetch({ fetch: fetcher, allowPrivateNetwork: isPrivateNetworkAccessAllowed });
    return validateOpenAiVideoCredential(input.values, guardedFetcher, signal);
  },
};
