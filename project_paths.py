"""Repository-relative defaults and portable dataset links."""

import os
from pathlib import Path
import shutil
import warnings


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts):
    """Return an absolute path under the repository, independent of the cwd."""
    return str(PROJECT_ROOT.joinpath(*parts))


def link_or_copy(source, destination):
    """Create an absolute symlink, or copy on Windows if symlinks are unavailable.

    Existing destinations are rejected, including broken links, to avoid silently
    reusing stale subsets or overwriting user data.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError as error:
        # Windows: access denied, unsupported filesystem, or missing privilege.
        if os.name != "nt" or getattr(error, "winerror", None) not in (5, 50, 1314):
            raise
        warnings.warn(
            f"Cannot create symbolic link ({error}); copying {source} to "
            f"{destination} instead. This uses additional disk space.",
            RuntimeWarning,
            stacklevel=2,
        )
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            # Exclusive creation also protects against a destination appearing
            # between the existence check and the fallback.
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copystat(source, destination)
    return str(destination)
