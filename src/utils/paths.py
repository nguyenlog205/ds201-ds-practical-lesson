from pathlib import Path
from functools import lru_cache

@lru_cache
def get_project_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return cur.parents[-1]

def project_path(*parts: str) -> Path:
    return get_project_root().joinpath(*parts)
