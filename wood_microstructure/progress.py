
"""Module for handling progress bars in the CLI."""
import atexit
from contextlib import contextmanager

from rich.progress import (BarColumn, DownloadColumn, MofNCompleteColumn,
                           Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn,
                           TransferSpeedColumn)


class PoochDownloadProgressBar:
    def __init__(self):
        self.progress = Progress(
            '[progress.description]{task.description}',
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )
        self.total = None
        self.task_id = None

    def update(self, n_bytes):
        if self.task_id is None:
            self.progress.start()
            self.task_id = self.progress.add_task('Downloading...', total=self.total)
        self.progress.update(self.task_id, advance=n_bytes)

    def close(self):
        self.progress.stop()

    def reset(self):
        self.progress.remove_task(self.task_id)
        self.task_id = None


ACTIVE_PROGRESS: Progress = Progress(
    SpinnerColumn(),
    TextColumn('[progress.description]{task.description}'),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
atexit.register(ACTIVE_PROGRESS.stop)

PROGRESS_BAR_LEVEL = 0

def set_progress_bar_level(level: int):
    """Set the global progress bar level."""
    global PROGRESS_BAR_LEVEL
    PROGRESS_BAR_LEVEL = level

@contextmanager
def progress_bar_level_inc(clean_tasks: bool = True):
    """Context manager to increase the progress bar level."""
    global PROGRESS_BAR_LEVEL
    PROGRESS_BAR_LEVEL += 1
    try:
        yield
    finally:
        if clean_tasks:
            progress_clean_tasks()
        PROGRESS_BAR_LEVEL -= 1

def progress_bar(
        iterable, total=None,
        description=None, **kwargs
    ):
    """Create a progress bar using rich."""

    ACTIVE_PROGRESS.start()

    if not total:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    kwargs['total'] = total

    description = '| ' * PROGRESS_BAR_LEVEL + (description or 'Working')

    return ACTIVE_PROGRESS.track(iterable, description=description, **kwargs)

def progress_clean_tasks():
    """Cleanup the progress bar."""
    for task in ACTIVE_PROGRESS.tasks:
        if task.completed == task.total or task.total is None:
            ACTIVE_PROGRESS.remove_task(task.id)
