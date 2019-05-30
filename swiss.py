# -*- encoding: utf-8 -*- python3
"""Solver for swiss pairings."""

import argparse
import contextlib
import enum
import fractions
import itertools
import math
import os
import random
import sys
from typing import List, Optional, Tuple

import blitzstein_diaconis
import elkai
import numpy as np
import player as player_lib
import sheet_manager

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

BYE = player_lib.Player('noreply', 'BYE', fractions.Fraction(0), 0, ())
EFFECTIVE_INFINITY = 1 << 20
FLAGS = None  # Parsing the flags needs to happen in main.
HUB_COST = 1

Pairings = List[Tuple[player_lib.Player, player_lib.Player]]


def Odd(n):
  return n % 2 == 1


def Even(n):
  return not Odd(n)


def Lcm(a, b):
  """Compute the lowest common multiple."""
  return a * b // math.gcd(a, b)


def SSE(pairings):
  """Returns the sum of squared error (SSE) of pairings."""
  return sum((p.score - q.score)**2 for (p, q) in pairings)


def SplitOnce(pairings: Pairings) -> Tuple[Pairings, Pairings]:
  """Split the pairings loop into two loops at the best crossover point."""
  best_split = (pairings, [])
  best_loss = SSE(pairings)
  for i in range(len(pairings)):
    for j in range(i, len(pairings)):
      # First check if the swap is valid.
      if (pairings[i][0].id in pairings[j][1].opponents or
          pairings[j][0].id in pairings[i][1].opponents or
          pairings[i][1].id in pairings[j][0].opponents or
          pairings[j][1].id in pairings[i][0].opponents):
        continue
      left = pairings[j + 1:] + pairings[:i]
      right = pairings[i + 1:j]
      left.append((pairings[i][0], pairings[j][1]))
      right.append((pairings[j][0], pairings[i][1]))
      if SSE(left + right) < best_loss:
        best_loss = SSE(left + right)
        best_split = (left, right)
  if best_split[1]:
    print(f'Found a split that improves loss by {SSE(pairings) - best_loss}.')
  return best_split


def SplitAll(pairings: Pairings) -> Pairings:
  """Recursively split the pairings as long as improvements are found."""
  left, right = SplitOnce(pairings)
  if left == pairings:
    return left
  return SplitAll(left) + SplitAll(right)


def PrintPairings(pairings, stream=sys.stdout):
  """Print a pretty table of the model to the given stream."""
  my_pairings = sorted(
      pairings, key=lambda t: (t[0].score, t[1].score, t), reverse=True)
  with contextlib.redirect_stdout(stream):
    for (p, q) in my_pairings:
      # 7 + 7 + 28 + 28 + 4 spaces + "vs." (3) = 77
      p_score = f'({p.score})'
      q_score = f'({q.score})'
      line = f'{p_score:>7} {p.name:>28} vs. {q.name:<28} {q_score:>7}'
      if abs(p.score - q.score) > 0:
        if stream.isatty():
          line = f'\033[1m{line}\033[0m'
      print(line)
    print()
    loss = SSE(pairings)
    print(f'Loss: {loss!s}')
    rmse = math.sqrt(SSE(pairings) / len(pairings))
    print(f'Root Mean Squared Error (per match): {rmse:.4f}')


class NodeType(enum.Enum):
  SINGLE = 1
  DOUBLE = 2
  HUB = 3


class Pairer(object):
  """Manages pairing a cycle of a league."""

  def __init__(self, players: List[player_lib.Player]):
    self.players = players
    self.players_by_id = {player.id: player for player in players}
    self.bye = None
    self.lcm = 1
    for d in set(p.score.denominator for p in self.players):
      self.lcm = Lcm(self.lcm, d)

  def GiveBye(self) -> Optional[player_lib.Player]:
    """Give a player a bye and return that player."""
    if Odd(sum(p.requested_matches for p in self.players)):
      eligible_players = [
          p for p in self.players if p.requested_matches == 3
          if BYE.id not in p.opponents
      ]
      bye = min(eligible_players, key=lambda p: (p.score, random.random()))
      self.players.remove(bye)
      self.bye = bye._replace(requested_matches=bye.requested_matches - 1)
      self.players.append(self.bye)
      return self.bye

  def MakePairings(self, random_pairings=False) -> Pairings:
    """Make pairings — random in cycle 1, else TSP optimized."""
    if random_pairings:
      print('Random pairings')
      pairings = self.RandomPairings()
    else:
      pairings = self.TravellingSalesPairings()
      pairings = SplitAll(pairings)
    if self.bye:
      pairings.append((self.bye, BYE))
    return pairings

  def RandomPairings(self) -> Pairings:
    """Generate and return random pairings."""
    degree_sequence = sorted(p.requested_matches for p in self.players)
    edge_set = blitzstein_diaconis.ImportanceSampledBlitzsteinDiaconis(
        degree_sequence)
    pairings = []
    players_by_index = dict(zip(itertools.count(), self.players))
    for (i, j) in edge_set:
      pairings.append((players_by_index[i], players_by_index[j]))

    if self.bye:
      pairings.append((self.bye, BYE))
    return pairings

  def TravellingSalesPairings(self):
    """Compute optimal pairings with a travelling-salesman solver."""
    odd_players = list(p for p in self.players if Odd(p.requested_matches))
    # random.shuffle(odd_players)
    assert Even(len(odd_players))

    counter = itertools.count()
    tsp_nodes = {}
    for p in self.players:
      for _ in range(p.requested_matches // 2):
        tsp_nodes[next(counter)] = (p, NodeType.DOUBLE)
    for p in odd_players:
      tsp_nodes[next(counter)] = (p, NodeType.SINGLE)
      tsp_nodes[next(counter)] = (p, NodeType.HUB)

    n = len(tsp_nodes)
    weights = np.zeros((n, n), dtype=int)
    for i in range(n):
      for j in range(n):
        p, ptype = tsp_nodes[i]
        q, qtype = tsp_nodes[j]
        if (ptype, qtype) in (
            (NodeType.DOUBLE, NodeType.DOUBLE),
            (NodeType.SINGLE, NodeType.SINGLE),
            (NodeType.SINGLE, NodeType.DOUBLE),
            (NodeType.DOUBLE, NodeType.SINGLE),
        ):
          if p == q or p.id in q.opponents or q.id in p.opponents:
            weights[i, j] = EFFECTIVE_INFINITY
          else:
            weights[i, j] = (int(p.score * self.lcm) -
                             int(q.score * self.lcm))**2
        elif (ptype, qtype) in ((NodeType.HUB, NodeType.SINGLE),
                                (NodeType.SINGLE, NodeType.HUB)):
          if p == q:
            weights[i, j] = 0
          else:
            weights[i, j] = EFFECTIVE_INFINITY
        elif (ptype, qtype) in ((NodeType.HUB, NodeType.DOUBLE),
                                (NodeType.DOUBLE, NodeType.HUB)):
          weights[i, j] = EFFECTIVE_INFINITY
        elif (ptype, qtype) == (NodeType.HUB, NodeType.HUB):
          weights[i, j] = (HUB_COST * self.lcm)**2
        else:
          assert False, f'{p.id} {ptype} -- {q.id} {qtype}'

    pairings = []
    expected_matches = sum(p.requested_matches for p in self.players) // 2
    while len(pairings) < expected_matches:
      pairings = []
      tour = elkai.solve_int_matrix(weights)
      for out, in_ in zip(tour, tour[1:] + [tour[0]]):
        p, ptype = tsp_nodes[out]
        q, qtype = tsp_nodes[in_]
        if (ptype, qtype) in (
            (NodeType.DOUBLE, NodeType.DOUBLE),
            (NodeType.SINGLE, NodeType.SINGLE),
            (NodeType.SINGLE, NodeType.DOUBLE),
            (NodeType.DOUBLE, NodeType.SINGLE),
        ):
          if (p, q) in pairings or (q, p) in pairings:
            weights[out, in_] = EFFECTIVE_INFINITY
            weights[in_, out] = EFFECTIVE_INFINITY
            o = tsp_nodes[out]
            i = tsp_nodes[in_]
            print(f'Nixing repeat match: {i[0].id}–{o[0].id}.')
            break
          else:
            pairings.append((p, q))
    return pairings


def Main():
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  sheet = sheet_manager.SheetManager(FLAGS.set_code, FLAGS.cycle)
  pairer = Pairer(sheet.GetPlayers())
  pairer.GiveBye()
  pairings = pairer.MakePairings(random_pairings=FLAGS.cycle in (1,))
  PrintPairings(pairings)
  try:
    os.mkdir('pairings')
  except FileExistsError:
    pass
  with open(f'pairings/pairings-{FLAGS.set_code}{FLAGS.cycle}.txt',
            'w') as output:
    PrintPairings(pairings, stream=output)

  if FLAGS.write:
    sheet.Writeback(sorted(pairings))


if __name__ == '__main__':
  FLAGS = flags.parse_args(sys.argv[1:])
  Main()
