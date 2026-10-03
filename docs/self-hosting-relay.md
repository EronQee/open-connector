# Self-Hosting Runbook: Relay Providers on nibrun

This clone is a working copy of the fork `EronQee/open-connector`, branch `relay-providers`, which adds two custom providers for OpenAI-protocol API relay services (中转站) that are not in the upstream catalog:

- `src/providers/openai_compatible/` — text (`chat/completions`, Responses API) and image generation/editing against a user-configured base URL. Covers lens (`https://lensapi.cn`), magic666 (`https://magic666.top`), qiuqiutoken (`https://api.qiuqiutoken.net/v1`), and any future OpenAI-shaped relay. A bare-host base URL gets `/v1` appended automatically.
- `src/providers/openai_video/` — asynchronous video generation tasks. Connection field `protocol` selects the task protocol: `openai` (default, `POST /v1/videos`, used by lens, magic666 MiniMax, qiuqiutoken Sora) or `seedance` (`POST /v1/video/generations`, Doubao Seedance relays). Protocol-specific request fields (MiniMax `content[]`, Seedance `metadata`, watermark flags) go through the documented `extraBody` passthrough.

Video bytes do NOT flow through the gateway by default: `get_video` returns the relay's `videoUrl` and the caller downloads the file to its own storage. The opt-in `download: true` input transfers the file into the connector's transit-file storage, only for relays whose download URL requires the API key or expires quickly.

## Deploy chain (every provider change)

The binary cannot be compiled on this Windows machine (the build requires Bun on Linux), so GitHub Actions builds it and WSL runs the deploy:

```sh
# 1. Commit provider changes and push to the fork (remote "fork").
git push fork relay-providers

# 2. Build linux-x64 on GitHub Actions (free ubuntu runner).
gh workflow run build-linux-x64.yml --repo EronQee/open-connector --ref relay-providers

# 3. Wait, then download the artifact into dist-deploy/.
gh run download <run-id> --repo EronQee/open-connector --name binary-linux-x64 --dir dist-deploy

# 4. Copy into WSL and redeploy the EXISTING nibrun app (in-place binary
#    replace; secrets, connections and the SQLite credential store in
#    /app/data survive every deploy).
wsl -e sh -c 'cp /mnt/d/Tools/open-connector/dist-deploy/open-connector-linux-x64 ~/open-connector-linux-x64 && chmod +x ~/open-connector-linux-x64 && ~/.local/bin/nib run ~/open-connector-linux-x64 --app open-connector'
```

## Environment facts (do not re-derive)

- `nib` CLI only ships darwin/linux binaries; it is installed inside WSL Ubuntu at `/home/administrator/.local/bin/nib`. `nib login` is a device flow that needs a human browser approval; `nib apps list` checks sign-in state.
- The live deployment is the app `open-connector`, slug `open-connector-79nk04`, URL `https://open-connector-79nk04.nibrun.app` (nibrun free tier: 1 vCPU / 256 MiB, sleeps after 5 minutes idle). `LAZY_SCHEMAS` is enabled to fit 256 MiB.
- The fork's default branch is `relay-providers` (changed from `main` so `build-linux-x64.yml` is workflow_dispatch-able; GitHub requires the workflow file on the default branch). The upstream `build-binary.yml` pins Blacksmith runner labels that queue forever on a fork without a Blacksmith account — do not dispatch it, use `build-linux-x64.yml`.
- Local git config `core.autocrlf=false` was set deliberately. With the Windows default `true`, `oxfmt` rewrites line endings across the whole tree and marks ~7000 files as modified. Do not re-enable it.
- `npx vitest run src/providers/provider-source-guards.clone-baseline.test.ts` fails on this Windows checkout even on pristine upstream code (path-separator/EOL sensitivity in the test itself); Linux CI is unaffected. Treat its output here as noise.
- New provider code must pass the shared-owners rules in AGENTS.md above; the two providers reuse `requiredInputString` / `requiredResponseRecord` / `optionalStringArray` / `runProviderRequest` / `assertPublicHttpUrl` from the shared runtime instead of local copies, and `openai_video` imports the base-URL normalizer from `openai_compatible/runtime.ts` (sibling import, one owner).

## Verify a deploy

```sh
curl https://open-connector-79nk04.nibrun.app/health          # -> {"ok":true}
oo search openai_compatible                                   # actions listed in the catalog
oo search openai_video
```

Connecting credentials happens in the console UI (Providers → OpenAI-Compatible Relay / Video Relay → baseUrl + apiKey, video also selects protocol). After that, agents use the standard `oo search → schema → run` flow.
