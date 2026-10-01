// A template remounts on every navigation, unlike the layout, so each route fades in.
// Opacity only: nothing moves, so the page never shifts while it appears.
export default function Template({ children }: { children: React.ReactNode }) {
  return <div className="route-fade">{children}</div>;
}
