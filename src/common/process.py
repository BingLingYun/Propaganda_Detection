"""Process helpers that keep optional runtime niceties from breaking scripts."""

from __future__ import annotations


def set_process_title(title: str) -> None:
    """Set the process title when ``setproctitle`` is installed."""
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)
