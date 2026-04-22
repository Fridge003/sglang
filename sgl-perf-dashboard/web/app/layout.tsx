import type { Metadata } from "next";
import Link from "next/link";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata: Metadata = {
  title: "sglang Perf Dashboard",
  description: "GB200 nightly benchmark regression tracking",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`dark ${GeistSans.variable} ${GeistMono.variable}`}
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <div className="pointer-events-none fixed inset-x-0 top-0 h-96 bg-gradient-to-b from-primary/[0.03] to-transparent" />
        <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur-md">
          <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
            <Link
              href="/"
              className="flex items-center gap-2 text-[15px] font-semibold tracking-tight"
            >
              <span className="inline-block h-2 w-2 rounded-full bg-primary" aria-hidden />
              sglang
              <span className="font-normal text-muted-foreground">/ GB200 perf</span>
            </Link>
            <nav className="flex items-center gap-5 text-[13px] text-muted-foreground">
              <Link href="/" className="transition hover:text-foreground">Home</Link>
              <Link href="/runs" className="transition hover:text-foreground">Runs</Link>
              <Link href="/configs" className="transition hover:text-foreground">Configs</Link>
              <Link href="/regressions" className="transition hover:text-foreground">Regressions</Link>
            </nav>
          </div>
        </header>
        <main className="relative mx-auto max-w-7xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
