"use client";

import type { ReactNode } from "react";
import { ProcessedDataProvider } from "@/lib/processed-data-context";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return <ProcessedDataProvider>{children}</ProcessedDataProvider>;
}
