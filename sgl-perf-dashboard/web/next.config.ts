import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Rewrite /api/* to the backend so the browser hits a same-origin URL.
  async rewrites() {
    const apiOrigin = process.env.API_INTERNAL_URL ?? "http://dashboard-api:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
