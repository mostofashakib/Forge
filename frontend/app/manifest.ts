import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Forge",
    short_name: "Forge",
    description:
      "Convert enterprise workflow specifications into reinforcement learning environments.",
    start_url: "/environments/new",
    display: "standalone",
    background_color: "#f5f5fb",
    theme_color: "#3730a3",
    icons: [
      { src: "/favicon.ico", sizes: "any", type: "image/x-icon" },
    ],
  };
}
