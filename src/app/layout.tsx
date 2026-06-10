import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BayFlood | Bay Area Flood Risk",
  description:
    "Estimate Bay Area flood exposure using FEMA zones, USGS elevation, and NOAA rainfall data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
