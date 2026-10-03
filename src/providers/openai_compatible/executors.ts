import type {
  CredentialValidationResult,
  CredentialValidators,
  ExecutionContext,
  ProviderExecutors,
} from "../../core/types.ts";
import type { OpenAiCompatibleActionContext } from "./runtime.ts";

import { isPrivateNetworkAccessAllowed } from "../../core/request.ts";
import { createProviderFetch, defineProviderExecutors, requireCustomCredential } from "../provider-runtime.ts";
import {
  createOpenAiCompatibleContext,
  openAiCompatibleActionHandlers,
  validateOpenAiCompatibleCredential,
} from "./runtime.ts";

const service = "openai_compatible";

export const executors: ProviderExecutors = defineProviderExecutors<OpenAiCompatibleActionContext>({
  service,
  handlers: openAiCompatibleActionHandlers,
  allowPrivateNetwork: isPrivateNetworkAccessAllowed,
  async createContext(context: ExecutionContext, fetcher: typeof fetch): Promise<OpenAiCompatibleActionContext> {
    const credential = await requireCustomCredential(context, service);
    return createOpenAiCompatibleContext(credential.values, fetcher, context.signal);
  },
  fallbackMessage: "unknown openai_compatible action",
});

export const credentialValidators: CredentialValidators = {
  customCredential(input, { fetcher, signal }): Promise<CredentialValidationResult> {
    const guardedFetcher = createProviderFetch({ fetch: fetcher, allowPrivateNetwork: isPrivateNetworkAccessAllowed });
    return validateOpenAiCompatibleCredential(input.values, guardedFetcher, signal);
  },
};
