"""The package's readme: the workspace's README.md, shared by both packages.

Hatchling takes a readme only from within the package's directory. So this hook
reads the workspace's, two directories up, or, in an sdist, the copy it holds.
"""

from __future__ import annotations

from pathlib import Path

from hatchling.metadata.plugin.interface import MetadataHookInterface


class ReadmeHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        root = Path(self.root)
        # In an sdist, its own copy. In the workspace, the README two directories up.
        for readme in (root / "README.md", root.parent.parent / "README.md"):
            if readme.is_file():
                metadata["readme"] = {
                    "content-type": "text/markdown",
                    "text": readme.read_text(encoding="utf-8"),
                }
                return
        raise FileNotFoundError("README.md, in the package or two directories up")
