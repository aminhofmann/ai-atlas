"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/", label: "Today" },
  { href: "/path", label: "Path" },
  { href: "/review", label: "Review" },
  { href: "/graph", label: "Map" },
];

/** Reads the URL, so the layout renders it inside <Suspense> (Cache Components). */
export function ActiveNav() {
  return <Nav pathname={usePathname()} />;
}

export function Nav({ pathname }: { pathname: string | null }) {
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur">
      <div className="mx-auto flex h-13 max-w-6xl items-center gap-8 px-5">
        <Link href="/" className="text-[15px] font-semibold tracking-tight">
          AI Atlas
        </Link>
        <nav className="flex gap-5 text-sm">
          {ITEMS.map((it) => {
            const active =
              pathname !== null && (it.href === "/" ? pathname === "/" : pathname.startsWith(it.href));
            return (
              <Link
                key={it.href}
                href={it.href}
                aria-current={active ? "page" : undefined}
                className={`py-4 transition-colors ${
                  active ? "text-ink shadow-[inset_0_-2px_0_var(--accent)]" : "text-ink-3 hover:text-ink"
                }`}
              >
                {it.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
