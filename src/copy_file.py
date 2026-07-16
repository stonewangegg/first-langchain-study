"""Small utility that copies a single file into a target folder."""

import shutil
from pathlib import Path
from typing import Optional, Union


def copy_file_to_folder(
    source: str | Path,
    target_folder: str | Path,
    new_filename: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Copy ``source`` into ``target_folder``.

    Args:
        source: Path to the file to copy.
        target_folder: Destination directory. Created (with parents) if missing.
        new_filename: Optional new name for the copied file. If ``None`` the
            original basename is kept.
        overwrite: If ``True``, an existing file with the same name will be
            replaced. Defaults to ``False`` to avoid accidental overwrites.

    Returns:
        The :class:`pathlib.Path` of the newly created file.

    Raises:
        FileNotFoundError: If ``source`` does not exist or is not a file.
        NotADirectoryError: If ``target_folder`` exists but is not a directory.
        FileExistsError: If a file with the target name already exists and
            ``overwrite`` is ``False``.
    """
    src_path = Path(source)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    dst_folder = Path(target_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    if not dst_folder.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {dst_folder}")

    dst_path = dst_folder / (new_filename or src_path.name)
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {dst_path}")

    # ``shutil.copy2`` preserves metadata (mtime, atime) in addition to the
    # file contents, which is usually what you want for a simple copy.
    shutil.copy2(src_path, dst_path)
    return dst_path


if __name__ == "__main__":
    # Minimal self-demo: create a temp file, copy it, then read it back.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        source = tmp / "hello.txt"
        source.write_text("Hello, world!", encoding="utf-8")

        destination = copy_file_to_folder(source, tmp / "copied")
        print(f"Copied to: {destination}")
        print(f"Contents : {destination.read_text(encoding='utf-8')}")