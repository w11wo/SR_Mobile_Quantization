import logging
import os
import os.path as osp
import numpy as np
from shutil import get_terminal_size
import sys
import time


def logger(name, filepath, resume=False):
    dir_path = osp.dirname(filepath)
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
    # if osp.exists(filepath) and resume==False:
    #     os.remove(filepath)

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s |[%(lineno)03d]%(filename)-11s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_hd = logging.StreamHandler()
    stream_hd.setFormatter(formatter)
    lg.addHandler(stream_hd)

    file_hd = logging.FileHandler(filepath)
    file_hd.setFormatter(formatter)
    lg.addHandler(file_hd)

    return lg


class ProgressBar(object):
    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                "terminal width is too small ({}), please consider widen the terminal for better progressbar visualization".format(
                    terminal_width
                )
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(
                "[{}] 0/{}, elapsed: 0s, ETA:\n{}\n".format(
                    " " * self.bar_width, self.task_num, "Prepare..."
                )
            )
        else:
            sys.stdout.write("completed:0, elapsed: 0s")
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg="Validation..."):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = ">" * mark_width + "-" * (self.bar_width - mark_width)
            sys.stdout.write("\033[2F")
            sys.stdout.write("\033[J")
            sys.stdout.write(
                "[{}] {}/{}, {:.1f} task/s elapsed: {}s, ETA: {:5}s\n{}\n".format(
                    bar_chars,
                    self.completed,
                    self.task_num,
                    fps,
                    int(elapsed + 0.5),
                    eta,
                    msg,
                )
            )
        else:
            sys.stdout.write(
                "completed: {}, elapsed: {}s, {:.1f} tasks/s".format(
                    self.completed, int(elapsed + 0.5), fps
                )
            )
        sys.stdout.flush()
