const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
};

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      ...JSON_HEADERS,
      ...corsHeaders(),
    },
  });
}

function normalizeName(name, fallback = "document.bin") {
  const safe = String(name || "").trim();
  return safe || fallback;
}

async function toMarkdown(env, name, blob) {
  if (!env?.AI?.toMarkdown) {
    throw new Error("Workers AI binding is not available");
  }

  const result = await env.AI.toMarkdown([
    {
      name: normalizeName(name),
      blob,
    },
  ]);

  const first = Array.isArray(result) ? result[0] : result;
  if (!first) return "";
  if (typeof first === "string") return first;
  return first.data || first.markdown || "";
}

function inferNameFromUrl(url, contentType) {
  try {
    const parsed = new URL(url);
    const pathname = parsed.pathname || "";
    const leaf = pathname.split("/").filter(Boolean).pop();
    if (leaf) return leaf;
  } catch {
    // Ignore malformed URLs here and fall back below.
  }

  if (contentType.includes("pdf")) return "document.pdf";
  if (contentType.includes("png")) return "image.png";
  if (contentType.includes("jpeg") || contentType.includes("jpg")) return "image.jpg";
  if (contentType.includes("webp")) return "image.webp";
  if (contentType.includes("tiff") || contentType.includes("tif")) return "image.tiff";
  if (contentType.includes("html")) return "page.html";
  return "document.bin";
}

async function convertUploadedFile(request, env) {
  const form = await request.formData();
  const file = form.get("file");

  if (!(file instanceof File)) {
    return jsonResponse({ success: false, error: "Missing file field" }, 400);
  }

  const markdown = await toMarkdown(env, file.name, file);
  return jsonResponse({
    success: true,
    markdown,
    filename: file.name,
    size_bytes: new TextEncoder().encode(markdown).length,
  });
}

async function convertUrl(request, env) {
  const url = new URL(request.url);
  const source = url.searchParams.get("url");

  if (!source) {
    return jsonResponse({ success: false, error: "Missing url query parameter" }, 400);
  }

  if (!source.startsWith("http")) {
    return jsonResponse({ success: false, error: "URL must start with http" }, 400);
  }

  const response = await fetch(source, { redirect: "follow" });
  if (!response.ok) {
    return jsonResponse(
      { success: false, error: `Upstream fetch failed with ${response.status}` },
      400,
    );
  }

  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  const name = inferNameFromUrl(source, contentType);
  const blob = await response.blob();
  const markdown = await toMarkdown(env, name, blob);

  return jsonResponse({
    success: true,
    markdown,
    url: source,
  });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return jsonResponse({
        status: "ok",
        backend: "cloudflare-workers-ai",
        markdown_conversion: true,
      });
    }

    if (url.pathname === "/convert" && request.method === "POST") {
      return convertUploadedFile(request, env);
    }

    if (url.pathname === "/convert-url" && request.method === "POST") {
      return convertUrl(request, env);
    }

    return jsonResponse({ success: false, error: "Not found" }, 404);
  },
};
