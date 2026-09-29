"""Small, dependency-free release hygiene check."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "flux", ROOT / "qwen_image")
TEXT_SUFFIXES = {".py", ".toml", ".md", ".txt", ".yaml", ".yml"}
FORBIDDEN = ("/data/public/", "/mnt/cpfs/", "/mnt/oss/")


def main() -> None:
    failures: list[str] = []
    for source_root in SOURCE_ROOTS:
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in FORBIDDEN:
                if marker in text:
                    failures.append(f"{path.relative_to(ROOT)} contains {marker!r}")

    if failures:
        raise SystemExit("Release hygiene check failed:\n" + "\n".join(failures))
    print("Release hygiene check passed.")


if __name__ == "__main__":
    main()
