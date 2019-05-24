# -*- encoding: utf-8 -*- python3
"""Solver for swiss pairings."""

from __future__ import print_function

import argparse
import contextlib
import fractions
import itertools
import math
import random
import sys
from typing import List, Optional, Tuple

import elkai

import blitzstein_diaconis
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
EFFECTIVE_INFINITY = 1 << 26


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
      ((abs(a.score - b.score), (a, b)) for (a, b) in pairings), reverse=True)
  final_loss = 0
  with contextlib.redirect_stdout(stream):
    for mismatch, (a, b) in my_pairings:
      # 7 + 7 + 28 + 28 + 4 spaces + "vs." (3) = 77
      a_score = f'({a.score})'
      b_score = f'({b.score})'
      line = f'{a_score:>7} {a.name:>28} vs. {b.name:<28} {b_score:>7}'
      if mismatch and stream.isatty():
        final_loss += mismatch * lcm**2
        line = '\033[1m{}\033[0m'.format(line)
      print(line)
    print()
    print(f'Total loss over LCM²: {final_loss} / {lcm**2}')
    rmse = math.sqrt(final_loss / lcm**2 / len(pairings))
    print(f'Root Mean Squared Error (per match): {rmse:.4f}')


Pairings = List[Tuple[player_lib.Player, player_lib.Player]]

BYE = player_lib.Player('BYE', 'noreply', fractions.Fraction(0), 0, [])


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
          if BYE not in p.opponents
      ]
      bye = min(eligible_players, key=lambda p: (p.score, random.random()))
      self.players.remove(bye)
      self.bye = bye._replace(requested_matches=bye.requested_matches - 1)
      self.players.append(self.bye)
      return self.bye

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

  def Search(self, random_pairings=False) -> Pairings:
    """Constructs an SMT problem for pairings and optimizes it."""
    if random_pairings:
      print('Random pairings')
      return self.RandomPairings()

    # Ensure NoRepeatMatches( self.previous_pairings, self.reverse_players)
    # Define Metric MismatchSum(slots, self.scores)
    # Ensure Requested Matches self.requested_matches, self.reverse_players
    for d in set(p.score.denominator for p in self.players):
      self.lcm = Lcm(self.lcm, d)

    odd_players = list(p for p in self.players if Odd(p.requested_matches))
    random.shuffle(odd_players)
    assert Even(len(odd_players))
    counter = itertools.count()
    tsp_nodes = {}
    for p in self.players:
      for _ in range(p.requested_matches // 2):
        tsp_nodes[next(counter)] = (p, p)
    num_odd_players = len(odd_players)
    # for (a, b) in zip(odd_players[:num_odd_players // 2],
    #                   odd_players[num_odd_players // 2:]):
    for a in odd_players:
      tsp_nodes[next(counter)] = (a, None)
      tsp_nodes[next(counter)] = (None, a)

    weights = []  # type: List[List[int]]
    for (player_z, player_a) in tsp_nodes.values():
      row = []
      weights.append(row)
      for (player_b, player_c) in tsp_nodes.values():
        if player_a == player_b or player_z == player_c:
          row.append(EFFECTIVE_INFINITY)
        elif player_a is not None and player_b is not None:
          row.append((int(player_a.score * self.lcm) -
                      int(player_b.score * self.lcm))**2)
        elif player_a is None and player_b is None:
          row.append(0)
        else:
          # Dense node meets singleton.
          row.append(EFFECTIVE_INFINITY)
    tour = elkai.solve_int_matrix(weights)
    pairings = []
    for out, in_ in zip(tour, tour[1:] + [tour[0]]):
      a = tsp_nodes[out][1]
      b = tsp_nodes[in_][0]
      if a and b:
        pairings.append((a, b))
      ex = a.id if a else None
      en = b.id if b else None
      print(f'{ex}→{en}')
    if self.bye:
      pairings.append((self.bye, BYE))
    n = sum(p.requested_matches for p in self.players) // 2
    print(f'I have {len(pairings)} matches but I should have {n}.')
    return pairings


def Main():
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  sheet = sheet_manager.SheetManager(FLAGS.set_code, FLAGS.cycle)
  pairer = Pairer(sheet.GetPlayers())
  pairer.GiveBye()
  pairings = pairer.Search(random_pairings=FLAGS.cycle in (1,))
  PrintPairings(pairings, pairer.lcm)
  with open(f'pairings-{FLAGS.set_code}{FLAGS.cycle}.txt', 'w') as output:
    PrintPairings(pairings, pairer.lcm, stream=output)

  if FLAGS.write:
    sheet.Writeback(pairings)


if __name__ == '__main__':
  FLAGS = flags.parse_args(sys.argv[1:])
  Main()
