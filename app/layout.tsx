import type { Metadata } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";
import { Suspense } from "react";
import { ActiveNav, Nav } from "@/components/nav";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
// Long-form explanations read in a serif; UI stays in Geist.
const reading = Source_Serif_4({ variable: "--font-reading", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Atlas",
  description: "A personal study space for transformers, from attention to protein models.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${reading.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <Suspense fallback={<Nav pathname={null} />}>
          <ActiveNav />
        </Suspense>
        <main className="flex flex-1 flex-col">{children}</main>
      </body>
    </html>
  );
}
