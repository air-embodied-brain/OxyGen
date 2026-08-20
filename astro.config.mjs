import { defineConfig } from "astro/config";

const isGitHubPages = process.env.GITHUB_ACTIONS === "true";

export default defineConfig({
  site: "https://air-embodied-brain.github.io",
  base: isGitHubPages ? "/OxyGen" : "/",
  output: "static",
  build: {
    format: "directory",
  },
});
