# MarkItDown Cloudflare Worker

This folder contains the Cloudflare Workers AI backend used by the UI toggle.

## Local dev

```bash
cd cloudflare
bun install
bun run dev
```

## Deploy

```bash
cd cloudflare
bun run deploy
```

The Worker exposes:

- `GET /health`
- `POST /convert`
- `POST /convert-url?url=...`

The UI can point at the Worker URL when "Use Cloudflare" is enabled.
