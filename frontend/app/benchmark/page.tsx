import { redirect } from "next/navigation";

export default function BenchmarkIndexPage() {
  redirect("/benchmark/run");
}
