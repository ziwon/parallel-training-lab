export function bytes(value: number): string {
  if (value === 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(4, Math.max(0, Math.floor(Math.log2(value) / 10)));
  return `${(value / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
export const ms = (value: number) => value.toFixed(2);
export const count = (value: number) =>
  value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value.toLocaleString("en-US");
