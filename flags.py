"""Flag definitions for swiss.py."""
import argparse
import sys

flags = argparse.ArgumentParser(description='Calculate multi-swiss pairings.')
flags.add_argument(
    'set_code',
    metavar='XYZ',
    type=str,
    help='the set code for the pairings spreadsheet',
)
flags.add_argument(
    'cycle',
    metavar='n',
    type=int,
    help='the cycle number to pair',
)
flags.add_argument(
    '-w',
    '--write',
    action='store_true',
    help='write the pairings to the spreadsheet',
)
flags.add_argument(
    '-f',
    '--fetch',
    action='store_true',
    help='force a fetch from the sheet, overriding the 20 minute cache timeout',
)

FLAGS = {}


def Init():
  FLAGS.update(flags.parse_args(sys.argv[1:]))
