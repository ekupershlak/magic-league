#!/bin/env python3
# -*- encoding: utf-8 -*-
"""Solver for swiss pairings."""

from __future__ import print_function

import argparse
import collections
import fractions
import importlib
import itertools
import math
import operator
import pickle
import random
import sys
import time

import password
import z3

BYE = 'BYE'

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
    help='whether to write the pairings to the spreadsheet',
)
flags.add_argument(
    '-t',
    '--timeout',
    metavar='n',
    type=int,
    default=600,
    help='time limit in seconds',
)
FLAGS = flags.parse_args(sys.argv[1:])


def Odd(n):
  return n % 2 == 1


def Lcm(a, b):
  return a * b // math.gcd(a, b)


def Timeleft(deadline):
  return int(deadline - time.time() + 0.5)


def MakeSlots(n_players):
  """Creates output pairing variables."""
  slots = collections.defaultdict(dict)
  for n in range(n_players):
    for m in range(n_players):
      if n < m:
        slots[n][m] = z3.Bool('m_{},{}'.format(n, m))
  return slots


def PopCount(bools, n):
  return z3.PbEq(list(zip(bools, itertools.repeat(1))), n)


def RequestedMatches(slots, requested_matches, reverse_players):
  """Guarantees players get their requested number of matches.

  Args:
    slots: slot variables
    requested_matches: the number of matches each player has requested
    reverse_players: the reverse_players dict
  Yields:
    Terms over slots (to be added to a Solver) that guarantees players have
    their requested number of matches.
  """

  n_players = len(slots) + 1
  order = sorted(
      range(n_players), key=lambda n: requested_matches[n], reverse=True)
  args = [(n, reverse_players[n], requested_matches[n], n_players)
          for n in order]
  for argtuple in args:
    yield RequestedMatchesForOnePlayer(argtuple)


def RequestedMatchesForOnePlayer(args):
  """Terms for asserting players play their requested number of matches."""
  n, name, requested, n_players = args
  print(name, 'requests', requested, 'matches')
  slots = MakeSlots(n_players)
  n_adjacency = []
  for m in range(n_players):
    if n < m:
      n_adjacency.append(slots[n][m])
    elif n > m:
      n_adjacency.append(slots[m][n])
  return PopCount(n_adjacency, requested)


def NoRepeatMatches(slots, previous_pairings, reverse_players):
  for n, row in list(slots.items()):
    for m, _ in list(row.items()):
      if (reverse_players[n], reverse_players[m]) in previous_pairings:
        yield z3.Not(slots[n][m])


def MismatchSum(slots, scores):
  """Returns an IntSort term for sum of squared mismatch."""
  lcm = 1
  for d in set(score.denominator for score in scores.values()):
    lcm = Lcm(lcm, d)
  sq_terms = []
  for n, row in list(slots.items()):
    for m, slot in list(row.items()):
      if n < m:
        diff = (scores[m] - scores[n])**2
        assert int(diff * lcm**2) == diff * lcm**2
        sq_terms.append(z3.If(slot, int(diff * lcm**2), 0))
  return z3.Sum(sq_terms)


class Pairer(object):
  """Manages pairing a cycle of a league."""

  def __init__(self, set_code, cycle):
    self.set_code = set_code
    self.cycle = cycle

    (names_scores_matches, self.previous_pairings, _,
     self.byed_name) = self._Fetch()
    self.players = {
        name: id
        for (id, (name, _, _)) in zip(itertools.count(), names_scores_matches)
    }
    self.scores = {
        id: score
        for (id, (_, score, _)) in zip(itertools.count(), names_scores_matches)
    }
    self.requested_matches = {
        id: m
        for (id, (_, _, m)) in zip(itertools.count(), names_scores_matches)
    }
    self.lcm = 1
    for d in set(score.denominator for score in self.scores.values()):
      self.lcm = Lcm(self.lcm, d)

  @property
  def reverse_players(self):
    return {number: name for name, number in list(self.players.items())}

  @property
  def player_scores(self):
    return {self.reverse_players[id]: score
            for (id, score) in list(self.scores.items())}

  def Search(self, seconds=3600, random_pairings=False):
    """Constructs an SMT problem for pairings and optimizes it."""
    if random_pairings:
      degree_sequence = [d for (_, d) in sorted(self.requested_matches.items())]
      edge_set = ImportanceSampledBlitzsteinDiaconis(degree_sequence)
      pairings = []
      if self.byed_name:
        pairings.append((self.byed_name, BYE))
      for (i, j) in edge_set:
        pairings.append((self.reverse_players[i], self.reverse_players[j]))
      print('Random pairings')
      print(pairings)
      return pairings

    deadline = time.time() + seconds
    s = z3.Optimize()
    slots = MakeSlots(len(self.players))
    for term in NoRepeatMatches(slots, self.previous_pairings,
                                self.reverse_players):
      s.add(term)

    metric = MismatchSum(slots, self.scores)
    for term in RequestedMatches(slots, self.requested_matches,
                                 self.reverse_players):
      s.add(term)
    print('lcm is', self.lcm)

    s.minimize(metric)
    s.set('timeout', Timeleft(deadline) * 1000)
    status = s.check()
    if status == z3.unsat:
      print()
      print('You dun goofed (Formula is unsatisfiable at any loss).')
      return
    elif status == z3.unknown:
      print('Time limit reached.')
    else:
      print('OPTIMAL!')

    model = s.model()
    final_loss = int(str(model.evaluate(metric)))
    self.PrintModel(slots, model, final_loss)
    with open(
        'pairings-{s.set_code}{s.cycle}.txt'.format(s=self), 'w') as output:
      self.PrintModel(slots, model, final_loss, stream=output)

    pairings = list(self.ModelPlayers(slots, model))
    if self.byed_name:
      pairings.append((self.byed_name, BYE))
    return pairings

  def Writeback(self, pairings):
    spreadsheet = self.GetSpreadsheet()
    ws_name = 'Cycle ' + str(self.cycle)
    output = spreadsheet.worksheet(ws_name)
    pairings_range = output.range('B2:C' + str(len(pairings) + 1))
    for cell, player in zip(pairings_range,
                            (player for row in pairings for player in row)):
      cell.value = player
    print('Writing to', ws_name)
    output.update_cells(pairings_range)

  def _Fetch(self, from_cache=True):
    """Fetches data from local file, falling back to the spreadsheet."""

    filename = '{s.set_code}-{s.cycle}'.format(s=self)
    if from_cache:
      try:
        return pickle.load(open(filename, 'rb'))
      except (IOError, EOFError):
        pass
    (names_scores_matches, previous_pairings, lcm,
     byed_name) = self._FetchFromSheet()
    pickle.dump((names_scores_matches, previous_pairings, lcm, byed_name),
                open(filename, 'wb'))

    return names_scores_matches, previous_pairings, None, byed_name

  def _FetchFromSheet(self):
    """Fetches data from the spreadsheet."""

    spreadsheet = self.GetSpreadsheet()
    standings = spreadsheet.worksheet('Standings')
    names = standings.col_values(2)[1:]
    wins, losses, draws = [
        [int(n) for n in standings.col_values(4 + c)[1:]] for c in range(3)
    ]
    scores = [
        fractions.Fraction(3 * w, 3 * (w + l + d))
        if w + l + d else fractions.Fraction(1, 2)
        for w, l, d in zip(wins, losses, draws)
    ]

    requested_matches = [
        int(s) for s in standings.col_values(9 + self.cycle - 1)[1:]
    ]
    names, requested_matches, scores = list(
        zip(*[(n, rm, s) for (n, rm, s) in zip(names, requested_matches, scores)
              if 0 < rm <= 3]))
    names = list(names)
    requested_matches = list(requested_matches)
    print(list(zip(names, requested_matches, scores)))

    previous_pairings = set()

    for i in range(1, self.cycle):
      cycle_sheet = spreadsheet.worksheet('Cycle {}'.format(i))
      a = cycle_sheet.col_values(2)[1:]
      b = cycle_sheet.col_values(3)[1:]

      previous_pairings |= set(zip(a, b))
      previous_pairings |= set(zip(b, a))

    if Odd(sum(requested_matches)):
      targetted_for_bye = 3
      candidates = [
          (i, name)
          for i, (name, n_requested) in enumerate(zip(names, requested_matches))
          if n_requested == targetted_for_bye and (name, BYE) not in
          previous_pairings
      ]
      byed_i, byed_name = random.choice(candidates)
      requested_matches[byed_i] -= 1
      print(byed_name, 'receives a bye.')
    else:
      byed_name = None

    names_scores_matches = list(zip(names, scores, requested_matches))
    random.shuffle(names_scores_matches)
    return names_scores_matches, previous_pairings, None, byed_name

  def GetSpreadsheet(self):
    return password.gc.open('magic-ny {} Sealed League'.format(self.set_code))

  def PrintModel(self, slots, model, final_loss, stream=sys.stdout):
    """Print a pretty table of the model to the given stream."""
    mismatches_and_lines = []
    for n, row in reversed(list(slots.items())):
      for m, playing in reversed(list(row.items())):
        if str(model.evaluate(playing)) == 'True':
          player = self.reverse_players[m]
          opponent = self.reverse_players[n]
          # 7 + 7 + 28 + 28 + 4 spaces + "vs." (3) = 77
          line = '{:>7} {:>28} vs. {:<28} {:>7}'.format(
              '({})'.format(self.scores[m]),
              player,
              opponent,
              '({})'.format(self.scores[n]),
          )
          mismatch = abs(self.scores[m] - self.scores[n])
          mismatches_and_lines.append((mismatch, line))
    total_matches = len(mismatches_and_lines)
    mismatches_and_lines.sort(key=operator.itemgetter(0))
    for mismatch, line in mismatches_and_lines:
      if mismatch and stream.isatty():
        line = '\033[1m{}\033[0m'.format(line)
      print(line, file=stream)
    print(file=stream)
    print(
        'Loss over LCM²: {} / {}'.format(final_loss, self.lcm**2), file=stream)
    print(
        'Root Mean Squared Error: {rmse:.4f}'.format(
            rmse=math.sqrt(final_loss / self.lcm**2 / total_matches)),
        file=stream)

  def ModelPlayers(self, slots, model):
    for n, row in reversed(list(slots.items())):
      for m, playing in reversed(list(row.items())):
        if str(model.evaluate(playing)) == 'True':
          yield (self.reverse_players[m], self.reverse_players[n])


def Main():
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  pairer = Pairer(FLAGS.set_code, FLAGS.cycle)
  pairings = pairer.Search(
      seconds=FLAGS.time_limit, random_pairings=FLAGS.cycle in (1,))

  if FLAGS.write_pairings:
    # Some aspect of the connection to the spreadsheet can go stale. Reload it
    # just before writing to make sure it's fresh.
    global password
    password = importlib.reload(password)  # pylint: disable=redefined-outer-name
    pairer.Writeback(pairings)


# Support functions for random pairings.
def Graphical(d):
  n = len(d)
  if any(di < 0 for di in d):
    return False
  d = [di for di in d if di > 0]
  d.sort(reverse=True)
  for k in range(1, n + 1):
    if sum(d[:k]) > k * (k - 1) + sum(min(k, di) for di in d[k:]):
      return False
  return True


def ArrayDecrement(indices, array):
  """Returns *copy of* `array`, having decremented each index in `indices`."""
  a = array[:]
  for i in indices:
    a[i] -= 1
  return a


def ImportanceSampledBlitzsteinDiaconis(d, n=100):
  """Sample from `n` graphs according to their probability of generation.

  See §8 of
  http://www.people.fas.harvard.edu/~blitz/BlitzsteinDiaconisGraphAlgorithm.pdf

  Args:
    d: a graphical degree sequence
    n: size of pool from which the final graph will be sampled by importance
  Returns:
    A graph with degree sequence `d`.
  Raises:
    ValueError: if d is not graphical
  """
  population = []
  weights = []
  for _ in range(n):
    e, c_sigma = BlitzsteinDiaconis(d)
    population.append(e)
    weights.append(1 / c_sigma)
  return random.choices(population, weights, k=1)[0]


def BlitzsteinDiaconis(d):
  """Generates a random graph with degree sequence `d`.

  See §4 of
  http://www.people.fas.harvard.edu/~blitz/BlitzsteinDiaconisGraphAlgorithm.pdf

  Args:
    d: a graphical degree sequence
  Returns:
    A graph with degree sequence `d`.
  Raises:
    ValueError: if d is not graphical
  """
  d = d[:]
  equivalence_class_size = 1
  likelihood = fractions.Fraction(1)
  e = set()
  if not Graphical(d):
    raise ValueError('{} is not graphical.'.format(d))
  while any(di > 0 for di in d):
    minimum = min(di for di in d if di > 0)
    i = d.index(minimum)
    equivalence_class_size *= minimum
    while d[i] > 0:
      candidates = {
          j
          for j in range(len(d))
          if j != i and tuple(sorted((i, j))) not in e and
          Graphical(ArrayDecrement((i, j), d))
      }
      selection = random.choices(
          list(candidates), [d[j] for j in candidates], k=1)[0]
      likelihood *= d[selection]
      likelihood /= sum(d[j] for j in candidates)
      e.add(tuple(sorted((i, selection))))
      d = ArrayDecrement((i, selection), d)
      assert Graphical(d)
  # print('c(Y) = {}'.format(equivalence_class_size))
  # print('σ(Y) = {}'.format(likelihood))
  return e, likelihood * equivalence_class_size


if __name__ == '__main__':
  Main()
