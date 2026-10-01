// The Forge mark: a spark struck off an anvil. Drawn on a 24px grid so it stays sharp at favicon size.
export const FORGE_MARK_PATHS = {
  anvil: "M2 11H21V14H16.5L15 16.5V18.5H17.5V21H6.5V18.5H9V16.5L7.5 14H5Z",
  spark: "M12 1.5L13.3 4.7L16.5 6L13.3 7.3L12 10.5L10.7 7.3L7.5 6L10.7 4.7Z",
  core: "M12 4.4L13.6 6L12 7.6L10.4 6Z",
};

export default function ForgeMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <path d={FORGE_MARK_PATHS.anvil} fill="var(--forge-ink, #1F1A14)" />
      <path d={FORGE_MARK_PATHS.spark} fill="var(--forge-spark, #F2541B)" />
      <path d={FORGE_MARK_PATHS.core} fill="var(--forge-core, #C9E94A)" />
    </svg>
  );
}
