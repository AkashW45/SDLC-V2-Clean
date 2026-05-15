import os
from pathlib import Path

ROOT = Path.cwd()

OUTPUT_FILE = "FULL_DEBUG_EXPORT.txt"

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next"
}

INCLUDE_EXTENSIONS = {
    ".py",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".env",
    ".txt",
    ".ini",
    ".html",
    ".swp"

}

MAX_FILE_SIZE = 500_000  # 500 KB


def should_skip(path: Path):
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    return False


with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write("=" * 100 + "\n")
    out.write("FULL PROJECT DEBUG EXPORT\n")
    out.write("=" * 100 + "\n\n")

    # ---------------------------------
    # PROJECT STRUCTURE
    # ---------------------------------

    out.write("\nPROJECT STRUCTURE\n")
    out.write("=" * 100 + "\n\n")

    for path in sorted(ROOT.rglob("*")):

        if should_skip(path):
            continue

        try:
            relative = path.relative_to(ROOT)

            depth = len(relative.parts)
            indent = "    " * (depth - 1)

            if path.is_dir():
                out.write(f"{indent}[DIR] {path.name}\n")
            else:
                out.write(f"{indent}{path.name}\n")

        except:
            pass

    # ---------------------------------
    # FILE CONTENTS
    # ---------------------------------

    out.write("\n\n")
    out.write("=" * 100 + "\n")
    out.write("FILE CONTENTS\n")
    out.write("=" * 100 + "\n\n")

    total_files = 0

    for file in sorted(ROOT.rglob("*")):

        if not file.is_file():
            continue

        if should_skip(file):
            continue

        if file.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue

        try:
            size = file.stat().st_size

            if size > MAX_FILE_SIZE:
                out.write("\n")
                out.write("=" * 100 + "\n")
                out.write(f"SKIPPED LARGE FILE: {file.relative_to(ROOT)}\n")
                out.write("=" * 100 + "\n\n")
                continue

            relative = file.relative_to(ROOT)

            out.write("\n")
            out.write("=" * 100 + "\n")
            out.write(f"FILE: {relative}\n")
            out.write("=" * 100 + "\n\n")

            content = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            out.write(content)
            out.write("\n\n")

            total_files += 1

        except Exception as e:
            out.write(f"\nERROR READING {file}: {e}\n")

    out.write("\n")
    out.write("=" * 100 + "\n")
    out.write(f"TOTAL FILES EXPORTED: {total_files}\n")
    out.write("=" * 100 + "\n")

print(f"\nDONE -> {OUTPUT_FILE}")