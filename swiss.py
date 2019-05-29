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
FLAGS = None  # Parsing the flags needs to happen in main.
EFFECTIVE_INFINITY = 1 << 20
HUB_COST = 1


def Odd(n):
  return n % 2 == 1


def Even(n):
  return not Odd(n)


def Lcm(a, b):
  """Compute the lowest common multiple."""
  return a * b // math.gcd(a, b)


def PrintPairings(pairings, lcm, stream=sys.stdout):
  """Print a pretty table of the model to the given stream."""
  my_pairings = sorted(
      pairings, key=lambda t: (t[0].score, t[1].score, t), reverse=True)
  final_loss = 0
  with contextlib.redirect_stdout(stream):
    for (a, b) in my_pairings:
      # 7 + 7 + 28 + 28 + 4 spaces + "vs." (3) = 77
      a_score = f'({a.score})'
      b_score = f'({b.score})'
      line = f'{a_score:>7} {a.name:>28} vs. {b.name:<28} {b_score:>7}'
      if abs(a.score - b.score) > 0:
        final_loss += abs(a.score - b.score) * lcm**2
        if stream.isatty():
          line = f'\033[1m{line}\033[0m'
      print(line)
    print()
    print(f'Total loss over LCM²: {final_loss} / {lcm**2}')
    rmse = math.sqrt(final_loss / lcm**2 / len(pairings))
    print(f'Root Mean Squared Error (per match): {rmse:.4f}')


Pairings = List[Tuple[player_lib.Player, player_lib.Player]]

BYE = player_lib.Player('noreply', 'BYE', fractions.Fraction(0), 0, ())


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

  def Search(self, random_pairings=False) -> Pairings:
    if random_pairings:
      print('Random pairings')
      pairings = self.RandomPairings()
    else:
      pairings = self.TravellingSalesPairings()
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
    for d in set(p.score.denominator for p in self.players):
      self.lcm = Lcm(self.lcm, d)

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
    tour = elkai.solve_int_matrix(weights)
    pairings = []
    for out, in_ in zip(tour, tour[1:] + [tour[0]]):
      p, ptype = tsp_nodes[out]
      q, qtype = tsp_nodes[in_]
      if (ptype, qtype) in (
          (NodeType.DOUBLE, NodeType.DOUBLE),
          (NodeType.SINGLE, NodeType.SINGLE),
          (NodeType.SINGLE, NodeType.DOUBLE),
          (NodeType.DOUBLE, NodeType.SINGLE),
      ):
        pairings.append((p, q))
    if self.bye:
      pairings.append((self.bye, BYE))
    return pairings


def Main():
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  sheet = sheet_manager.SheetManager(FLAGS.set_code, FLAGS.cycle)
  pairer = Pairer(sheet.GetPlayers())
  pairer.GiveBye()
  pairings = pairer.Search(random_pairings=FLAGS.cycle in (1,))
  PrintPairings(pairings, pairer.lcm)
  try:
    os.mkdir('pairings')
  except FileExistsError:
    pass
  with open(f'pairings/pairings-{FLAGS.set_code}{FLAGS.cycle}.txt',
            'w') as output:
    PrintPairings(pairings, pairer.lcm, stream=output)

  if FLAGS.write:
    sheet.Writeback(sorted(pairings))


if __name__ == '__main__':
  FLAGS = flags.parse_args(sys.argv[1:])
  Main()
