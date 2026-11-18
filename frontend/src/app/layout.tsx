import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vyrex AI — Vyrex Hyperion 3",
  description:
    "Vyrex AI powered by Vyrex Hyperion 3 — an independent conversational AI fine-tuned with 4-bit QLoRA and calibrated natural termination.",
  icons: {
    icon: "/vyrex-symbol.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const stored = localStorage.getItem('theme');
                if (stored === 'dark') {
                  document.documentElement.classList.add('dark');
                } else {
                  document.documentElement.classList.remove('dark');
                }
              } catch (_) {}
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-white text-gray-900 dark:bg-[#0d0d0d] dark:text-gray-100 antialiased selection:bg-red-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
