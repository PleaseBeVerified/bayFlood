import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";

const terrainPath = path.join(
  process.cwd(),
  "terrain",
  "runtime",
  "terrain-features-30m.tif",
);
const port = Number(process.env.TERRAIN_SERVER_PORT ?? 8_765);

if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error("TERRAIN_SERVER_PORT must be a valid TCP port.");
}

function parseRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header ?? "");
  if (!match || (!match[1] && !match[2])) return null;
  let start;
  let end;
  if (!match[1]) {
    const suffixLength = Number(match[2]);
    start = Math.max(0, size - suffixLength);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
  }
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    start >= size
  ) {
    return null;
  }
  return { start, end: Math.min(end, size - 1) };
}

const server = createServer(async (request, response) => {
  if (
    (request.method !== "GET" && request.method !== "HEAD") ||
    request.url !== "/terrain-features-30m.tif"
  ) {
    response.writeHead(404).end();
    return;
  }

  try {
    const details = await stat(terrainPath);
    const rangeHeader = request.headers.range;
    const range = parseRange(rangeHeader, details.size);
    if (rangeHeader && !range) {
      response.writeHead(416, {
        "Content-Range": `bytes */${details.size}`,
      }).end();
      return;
    }

    const headers = {
      "Accept-Ranges": "bytes",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "public, max-age=3600",
      "Content-Type": "image/tiff",
    };
    if (!range) {
      response.writeHead(200, {
        ...headers,
        "Content-Length": details.size,
      });
      if (request.method === "HEAD") response.end();
      else createReadStream(terrainPath).pipe(response);
      return;
    }

    response.writeHead(206, {
      ...headers,
      "Content-Length": range.end - range.start + 1,
      "Content-Range": `bytes ${range.start}-${range.end}/${details.size}`,
    });
    if (request.method === "HEAD") response.end();
    else createReadStream(terrainPath, range).pipe(response);
  } catch (error) {
    console.error(error);
    response.writeHead(500).end();
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(
    `Terrain COG range server: http://127.0.0.1:${port}/terrain-features-30m.tif`,
  );
});
