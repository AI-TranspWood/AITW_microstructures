"""Implement clock to time the execution of functions."""

import time
from collections import defaultdict
from functools import wraps


class ClockItem():
    def __init__(self, name: str):
        self.name = name
        self.tot_time = 0.0
        self.num_calls = 0
        self.sub_clocks: dict[str, ClockItem] = {}

    def report(self, all_total_time: int = 0, lvl: int = 0, lvl_sep: str = '| ') -> str:
        res = []
        name = self.name
        num_calls = self.num_calls
        tot_time = self.tot_time
        avg_time = (tot_time / num_calls * 1000) if num_calls > 0 else 0

        frc_string = ''
        if all_total_time > 0:
            frc_time = (tot_time / all_total_time * 100)
            frc_string = f'{frc_time:5.1f}% '

        name = f'{lvl_sep * lvl}{name}'

        res.append(
            f'{name:20s}   ({num_calls:>7d} CALLs):' +
            f'{frc_string}{tot_time:>13.4f} s  ({avg_time:>10.1f} ms/CALL)'
        )

        for sub_clock in self.sub_clocks.values():
            res.append(sub_clock.report(all_total_time, lvl + 1, lvl_sep))

        return '\n'.join(res)

class Clock():
    """Class to time the execution of functions.
    Usage:
    class MyClass(Clock):
        @Clock.register('clock_name')
        def my_function(self):
            pass
    """
    CLOCK_MARKER = '__clock__'

    def __init__(self, *args, **kwargs):
        self.clocks = {'total': ClockItem('total')}
        # self.clock_stack = defaultdict(int)
        self.clock_stack = 0
        self.clock_stack_mem = defaultdict(int)
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name):
        res = super().__getattribute__(name)
        if callable(res) and hasattr(res, Clock.CLOCK_MARKER) and not hasattr(res, '__wrapped__'):
            clock_lst = getattr(res, Clock.CLOCK_MARKER, None)
            if clock_lst is not None:
                clock_items: list[ClockItem] = []
                ptr = self.clocks
                for clock in clock_lst:
                    new = ptr.setdefault(clock, ClockItem(clock))
                    # ptr = self.clocks.setdefault(clock, defaultdict(int))
                    clock_items.append(new)
                    ptr = new.sub_clocks
                total_clock = self.clocks['total']
                @wraps(res)
                def wrapped(*args, **kwargs):

                    self.clock_stack += 1

                    start = time.time()
                    result = res(*args, **kwargs)
                    delta = time.time() - start

                    # When calling nested clock, remove the time spent in inner clocks from outer clocks
                    delta_min = delta - self.clock_stack_mem.pop(self.clock_stack, 0)
                    for clock_item in clock_items:
                        clock_item.tot_time += delta_min
                        clock_item.num_calls += 1
                    self.clock_stack -= 1

                    # Accumulate time spent in this level to subtract from outer clocks
                    self.clock_stack_mem[self.clock_stack] += delta
                    # Only the outermost clock counts towards total time
                    if self.clock_stack == 0:
                        total_clock.tot_time += delta
                        total_clock.num_calls += 1
                    return result
                wrapped.__wrapped__ = res
                return wrapped
        return res

    def report_clocks(self) -> str:
        res = ['Time report:']
        all_tot_time = self.clocks['total'].tot_time

        for clock in self.clocks.values():
            res.append(clock.report(all_tot_time))
            res.append('')

        return '\n'.join(res)

    @staticmethod
    def register(names: str | list[str]):
        if isinstance(names, str):
            names = [names]
        def decorator(func):
            setattr(func, Clock.CLOCK_MARKER, names)
            return func
        return decorator
