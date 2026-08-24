import { mkdir, rename, stat } from "node:fs/promises";
import { spawn } from "node:child_process";

const cacheDirectory = "data/cache/openfema";
const files = [
  {
    name: "NfipClaimsV3.parquet",
    url: "https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet",
    minimumBytes: 100_000_000,
  },
  {
    name: "NfipPoliciesV3.parquet",
    url: "https://www.fema.gov/about/reports-and-data/openfema/v3/NfipPoliciesV3.parquet",
    minimumBytes: 1_000_000_000,
  },
];

async function isComplete(path, minimumBytes) {
  const file = await stat(path).catch(() => null);
  return Boolean(file && file.size >= minimumBytes);
}

function download(url, partialPath) {
  return new Promise((resolve, reject) => {
    const child = spawn("curl", [
      "-L", "--fail", "--retry", "5", "--retry-all-errors", "-C", "-",
      "-o", partialPath, url,
    ], { stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code) => code === 0
      ? resolve()
      : reject(new Error(`curl exited with status ${code}`)));
  });
}

async function main() {
  await mkdir(cacheDirectory, { recursive: true });
  for (const file of files) {
    const finalPath = `${cacheDirectory}/${file.name}`;
    const partialPath = `${finalPath}.partial`;
    if (await isComplete(finalPath, file.minimumBytes)) {
      console.log(`Using cached ${finalPath}.`);
      continue;
    }
    console.log(`Downloading ${file.name}; the policy file is approximately 3.5 GB.`);
    await download(file.url, partialPath);
    if (!(await isComplete(partialPath, file.minimumBytes))) {
      throw new Error(`${file.name} download is unexpectedly small.`);
    }
    await rename(partialPath, finalPath);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
