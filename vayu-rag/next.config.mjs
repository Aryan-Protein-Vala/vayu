/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // In local dev the backend runs on localhost:8000, so proxy /api and /ws.
  // In deployment (Vercel), NEXT_PUBLIC_API_URL / NEXT_PUBLIC_WS_URL point
  // straight at the hosted backend, so no rewrites are needed.
  async rewrites() {
    if (process.env.NEXT_PUBLIC_API_URL) return []
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/ws/audio',
        destination: 'http://localhost:8000/ws/audio',
      },
    ]
  },
}

export default nextConfig