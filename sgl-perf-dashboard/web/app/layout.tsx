import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "sglang Perf Dashboard",
  description: "GB200 nightly benchmark regression tracking",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background font-sans antialiased">
        <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
            <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="text-lg">sglang</span>
              <span className="text-xs text-muted-foreground">/ GB200 perf</span>
            </Link>
            <nav className="flex items-center gap-6 text-sm text-muted-foreground">
              <Link href="/" className="hover:text-foreground">Home</Link>
              <Link href="/runs" className="hover:text-foreground">Runs</Link>
              <Link href="/configs" className="hover:text-foreground">Configs</Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
