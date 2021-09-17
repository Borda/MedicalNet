#!/usr/bin/env python


def load_lines(file_path):
    """Read file into a list of lines.

    Input
      file_path: file path

    Output
      lines: an array of lines
    """
    with open(file_path) as fio:
        lines = fio.read().splitlines()
    return lines
