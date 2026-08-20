import type { Metadata, Viewport } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'VĀYU — Voice RAG System',
  description: 'A voice-enabled retrieval system built for HH Goa 2026.',
  generator: 'VĀYU',
}

export const viewport: Viewport = {
  colorScheme: 'dark',
  themeColor: '#050607',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="bg-background" suppressHydrationWarning>
      <body className="antialiased">{children}</body>
    </html>
  )
}