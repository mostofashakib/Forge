import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Forge",
    short_name: "Forge",
    description:
      "Convert enterprise workflow specifications into reinforcement learning environments.",
    start_url: "/environments/new",
    display: "standalone",
    background_color: "#F1E9D8",
    theme_color: "#1F1A14",
    icons: [
      { src: "/icon.svg", sizes: "any", type: "image/svg+xml" },
    ],
  };
}
