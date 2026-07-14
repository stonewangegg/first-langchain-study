"""
"""

# Config the Built-in Virtual Filesystem Backend
import os
from pathlib import Path

from deepagents.backends import FilesystemBackend

from ..common_utils import uru_logger, FILE_DIR

FS_BACKEND = FilesystemBackend(root_dir=FILE_DIR, virtual_mode=True)

# Mount/copy skills into virtual filesystem
# 1. Get the absolute path of the directory where THIS tool script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Build the target path relative to THIS directory
# skill_target_path = os.path.join(CURRENT_DIR, "../skills/senior-industry-web-crawler/SKILL.md")
skill_target_folder = os.path.join(current_dir, "../skills")
# 3. (Optional but recommended) Normalize the path to remove the "../"
skill_target_folder = os.path.normpath(skill_target_folder)
# 4. copy each SKILL.md into the virtual filesystem (with recursive)
for file in Path(skill_target_folder).glob("**/*"):

    if not file.is_file():
            continue

    with open(file, "r", encoding="utf-8") as f:
        skill_content = f.read()

    file_relative_paht = "/"+str(file.relative_to(Path(skill_target_folder).parent))
    FS_BACKEND.write(
        file_relative_paht,
        skill_content
    )

    uru_logger.get_logger().info(f"Read skill file: {file_relative_paht}, write to virtual filesystem backend.")