import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  // Concept pages and the tutor read content/*.md at request time via fs, which
  // the bundler can't see — ship the folder with every route when hosted (Vercel).
  outputFileTracingIncludes: {
    "/**": ["./content/**/*"],
  },
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
