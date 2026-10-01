import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The synthetic task generator moved from /tasks. Old links still land on it.
  async redirects() {
    return [
      { source: "/tasks", destination: "/generator", permanent: true },
      { source: "/tasks/:batchId", destination: "/generator/:batchId", permanent: true },
    ];
  },
};

export default nextConfig;
