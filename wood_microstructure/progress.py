
"""Module for handling progress bars in the CLI."""
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
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


class RichMixin:
    pooc_progress_bar_cls = PoochDownloadProgressBar

    def __init__(self, *args, rich_live: Live = None, **kwargs):
        if rich_live is None:
            rich_live = Live(Group(), refresh_per_second=4)
        progress_group: Group = rich_live.renderable

        overall_progress = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        step_progress = Progress(
            SpinnerColumn(),
            TimeElapsedColumn(),
            TextColumn('[progress.description]{task.description}'),
            # BarColumn(),
            # MofNCompleteColumn(),
            # TimeRemainingColumn(),
        )
        current_step_progress = Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        current_group = Group(
            Panel(
                Group(step_progress, current_step_progress),
                title='Step Progress',
                expand=False,
            ),
            overall_progress,
        )
        progress_group.renderables.append(current_group)

        self.overall_progress = overall_progress
        self.step_progress = step_progress
        self.current_step_progress = current_step_progress
        # self.progress_group = progress_group
        self.current_step_id = None
        self.rich_live = rich_live

        super().__init__(*args, **kwargs)

    def track_step(self, iterable, total=None, description=None):
        """Get a progress bar for the given iterable."""
        if not total:
            try:
                total = len(iterable)
            except TypeError:
                total = None

        csp = self.current_step_progress

        task_id = csp.add_task(description or 'Processing...', total=total)
        csp.update(task_id, description=description, total=total)

        def new_iterable():
            for item in iterable:
                yield item
                csp.update(task_id, advance=1)
            csp.remove_task(task_id)

        return new_iterable()
