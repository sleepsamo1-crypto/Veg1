# ml/django_bootstrap.py
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def bootstrap_django(project_root: Path | None = None) -> str:
    """
    Robust Django bootstrap for standalone scripts.

    - Detect manage.py location (ROOT/manage.py or ROOT/backend/manage.py)
    - Parse DJANGO_SETTINGS_MODULE from manage.py
    - Insert manage.py dir into sys.path
    - Set env + django.setup()

    Returns the settings module string.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]  # .../D:\Veg

    project_root = Path(project_root).resolve()
    os.chdir(project_root)

    candidates = [
        project_root / "manage.py",
        project_root / "backend" / "manage.py",
    ]

    manage_dir = None
    settings_module = None

    for mp in candidates:
        if mp.exists():
            txt = mp.read_text(encoding="utf-8", errors="ignore")
            m = re.search(
                r"os\.environ\.setdefault\(\s*['\"]DJANGO_SETTINGS_MODULE['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
                txt,
            )
            manage_dir = mp.parent
            if m:
                settings_module = m.group(1)
            break

    if manage_dir is None:
        raise FileNotFoundError(
            f"Cannot find manage.py under {project_root} or {project_root / 'backend'}"
        )

    if not settings_module:
        # fallback (won't usually be needed if manage.py is normal)
        settings_module = "backend.settings"

    sys.path.insert(0, str(manage_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

    import django  # noqa: E402

    django.setup()

    return settings_module
