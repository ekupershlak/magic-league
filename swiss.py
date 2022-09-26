# python3
"""Solver for swiss pairings."""

import collections
import concurrent.futures
import contextlib
import difflib
import enum
import fractions
import itertools
import math
import multiprocessing
import os
import queue
import random
import sys
import threading
import time
import warnings

from typing import List, Optional, Tuple
from absl import app
from absl import flags

import blitzstein_diaconis
import elkai
import magic_sets
import numpy as np
import player as player_lib
import sheet_manager
import networkx as nx

BYE = player_lib.Player('#N/A', 'BYE', fractions.Fraction(0), 0, ())
DISCOURAGEMENT = 1
FLAGS = flags.FLAGS

Pairings = List[Tuple[player_lib.Player, player_lib.Player]]

flags.DEFINE_bool(
    'write', False, 'Write the pairings to the spreadsheet', short_name='w')
flags.DEFINE_bool(
    'tabprint', False, 'Write tab-separated names to stdout.', short_name='t')
flags.DEFINE_bool(
    'fetch',
    False,
    'Force a fetch from the sheet, overriding the 20 minute cache timeout.',
    short_name='f')


def Odd(n):
  return n % 2 == 1


def Rindex(lst, elt):
  """Returns the index of the rightmost occurrence of `elt` in `lst`."""
  return len(lst) - list(reversed(lst)).index(elt) - 1


def SSE(pairings: Pairings):
  """Returns the sum of squared error (SSE) of pairings."""
  return sum((p.score - q.score)**2 for (p, q) in pairings)


def ValidatePairings(pairings: Pairings, n: Optional[int] = None) -> None:
  """Raises an error if the pairings aren't valid.

  Args:
    pairings: The proposed pairings.
    n: The expected number of pairings.

  Raises:
    WrongNumberOfMatchesError: There were not `n` matches.
    DuplicateMatchError: If the proposed pairings contain a duplicate.
    RepeatMatchWarning: If the proposed pairings contain a match that occurred
        in a previous cycle.
    SelfMatchError: If a player is matched to themself.
  """
  if n is not None and len(pairings) != n:
    raise WrongNumberOfMatchesError(
        f'There are {len(pairings)} matches, but {n} were expected.')
  if len(set(tuple(sorted(match)) for match in pairings)) < len(pairings):
    # Duplicate matches
    matches = collections.Counter(tuple(sorted(match)) for match in pairings)
    dupes = []
    while matches:
      match, multiplicity = matches.most_common(1)[0]
      if multiplicity > 1:
        dupes.append(f'({match[0].id}, {match[1].id})')
        matches.pop(match)
      else:
        break
    if dupes:
      raise DuplicateMatchError(' '.join(dupes))
  for p, q in pairings:
    if p == q:
      raise SelfMatchError(p)
    if p.id in q.opponents or q.id in p.opponents:
      matches_ago = max(
          len(q.opponents) - Rindex(q.opponents, p.id),
          len(p.opponents) - Rindex(p.opponents, q.id))
      warnings.warn(f'{p.id} vs. {q.id} from {matches_ago} matches ago.',
                    RepeatMatchWarning)


def RoundTo(n, to):
  """Rounds `n` to the nearest increment of `to`."""
  return int(n / to + 0.5) * to


def PrintPairings(pairings, stream=sys.stdout):
  """Print a pretty table of the model to the given stream."""
  with contextlib.redirect_stdout(stream):
    for (p, q) in pairings:
      # 6 + 6 + 28 + 28 + 4 spaces + "vs." (3) = 75
      p_score = f'{float(p.score):.3f}'.lstrip('0')
      q_score = f'{float(q.score):.3f}'.lstrip('0')
      n_stars = min(
          5,
          max(
              0,
              RoundTo(5 + math.log(max(0.00001, abs(p.score - q.score)), 2),
                      0.5)))
      star_string = '⬥' * int(n_stars)
      if n_stars % 1 == 0.5:  # Exact float comparison!? Should be OK because we just rounded to an exact power of 2.
        star_string += '⬦'
      if n_stars > 2:
        if stream.isatty():
          star_string = f'\033[1m{star_string}\033[0m'
      line = f'({p_score:>4}) {p.name:>28} vs. {q.name:<28} ({q_score:>4}) {star_string}'
      print(line)
    print()
    rmse = math.sqrt(SSE(pairings) / len(pairings))
    print(f'Root Mean Squared Error (per match): {rmse:.4f}')


class Pairer(object):
  """Manages pairing a cycle of a league."""

  def __init__(self, players: List[player_lib.Player], sigma=0.0):
    self.sigma = sigma
    self.players = players
    self.players_by_id = {player.id: player for player in players}
    self.byed_player = None

  @property
  def correct_num_matches(self):
    """Returns the number of non-BYE matches that there *should* be."""
    return sum(player.requested_matches for player in self.players) // 2

  def GiveBye(self) -> Optional[player_lib.Player]:
    """Select a byed player if one is needed.

    If the total number of requested matches is odd, a bye is needed. Select a
    random 3-match-requester from among those with the lowest score. Mark that
    player as byed, decrease their requested matches, and return that player.
    It does NOT add a match representing the bye to any list of pairings.

    If the total number of requested matches is even, return None.

    Returns:
      The Player object of the player that got the bye.
    """
    if Odd(sum(p.requested_matches for p in self.players)):
      eligible_players = [
          p for p in self.players if p.requested_matches == 3
          if BYE.id not in p.opponents
      ]
      byed_player = min(
          eligible_players, key=lambda p: (p.score, random.random()))
      self.players.remove(byed_player)
      self.byed_player = byed_player._replace(
          requested_matches=byed_player.requested_matches - 1)
      self.players.append(self.byed_player)
      return self.byed_player

  def MakePairings(self, random_pairings=False) -> Pairings:
    """Make pairings — random or optimized depending on parameter."""
    if random_pairings:
      print('Random pairings')
      pairings = self.RandomPairings()
    else:
      print('Optimizing pairings')
      pairings = self.MaximumMatchingPairings()
    if self.byed_player:
      pairings.append((self.byed_player, BYE))
    return pairings

  def RandomPairings(self) -> Pairings:
    """Generate and return random pairings."""
    degree_sequence = [p.requested_matches for p in self.players]
    edge_set = blitzstein_diaconis.ImportanceSampledBlitzsteinDiaconis(
        degree_sequence)
    pairings = []
    players_by_index = dict(enumerate(self.players))
    for (i, j) in edge_set:
      pairings.append((players_by_index[i], players_by_index[j]))
    return pairings

  def MaximumMatchingPairings(self):
    """Compute optimal pairings with a maximum matching solver."""
    degree_sequence = [p.requested_matches for p in self.players]
    assert blitzstein_diaconis.Graphical(
        degree_sequence), 'Degree sequence is not graphical.'

    nodes = list(
        collections.Counter({p: p.requested_matches for p in self.players
                            }).elements())
    g = nx.Graph()
    for i in range(len(nodes)):
      p = nodes[i]
      for j in range(len(nodes)):
        q = nodes[j]
        if p == q:
          continue
        weight = (p.score - q.score + random.gauss(0, self.sigma))**2
        if p.id in q.opponents or q.id in p.opponents:
          factor = max(
              Rindex(q.opponents, p.id) / len(q.opponents),
              Rindex(p.opponents, q.id) / len(p.opponents))
          weight += DISCOURAGEMENT * factor
        g.add_edge(i, j, weight=weight)
    while True:
      m = nx.min_weight_matching(g)
      pairings = [(nodes[a], nodes[b]) for (a, b) in m]
      try:
        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          ValidatePairings(pairings)
        break
      except DuplicateMatchError:
        ordered_m = random.sample(list(m), k=len(m))
        seen = set()
        for (a, b) in ordered_m:
          if (nodes[a], nodes[b]) in seen or (nodes[b], nodes[a]) in seen:
            g.remove_edge(a, b)
            break
          seen.add((nodes[a], nodes[b]))
    return pairings


def OrderPairingsByTsp(pairings: Pairings) -> Pairings:
  """Sort the given pairings by minimal cost tour.

  After the pairings are determined (a completely independent step), this
  function uses a TSP reduction to sort the pairings such that there is a low
  "display diff" between adjacent matches on the printed list of pairings.

  In other words, it clusters match entries so players' matches appear near each
  other. It reduces players' need to Ctrl-F for their names and makes the list
  so very pretty.
  """
  pairings = pairings[:]
  random.shuffle(pairings)
  num_nodes = 2 * len(pairings) + 1
  weights = np.zeros((num_nodes, num_nodes), dtype=float)

  for alpha in range(len(pairings)):
    alpha_left = 2 * alpha + 1
    alpha_right = 2 * alpha + 2
    weights[alpha_left, alpha_right] = weights[alpha_right, alpha_left] = -1
    for beta in range(len(pairings)):
      if beta == alpha:
        continue
      beta_left = 2 * beta + 1
      beta_right = 2 * beta + 2
      # normal;normal
      # swapped;swapped
      weights[alpha_right, beta_left] = weights[alpha_left, beta_right] = (
          PairingTransitionCost(pairings[alpha], pairings[beta]))
      # normal;swapped
      # swapped;normal
      weights[alpha_right, beta_right] = weights[alpha_left, beta_left] = (
          PairingTransitionCost(pairings[alpha], pairings[beta][::-1]))
  tour = elkai.solve_float_matrix(weights)
  output_pairings = []
  for node in tour[1::2]:
    next_pairing = pairings[(node - 1) // 2]
    if node % 2 == 0:
      next_pairing = next_pairing[::-1]
    output_pairings.append(next_pairing)
  return output_pairings


def PairingTransitionCost(pairing_alpha, pairing_beta) -> float:
  left_cost = 1 - difflib.SequenceMatcher(
      a=pairing_alpha[0], b=pairing_beta[0]).ratio()
  right_cost = 1 - difflib.SequenceMatcher(
      a=pairing_alpha[1], b=pairing_beta[1]).ratio()
  return left_cost + right_cost


def Main(argv):
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  set_code, cycle = argv[1:]
  cycle = int(cycle)
  previous_set_code = list(magic_sets.names.keys())[-2]
  sheet = sheet_manager.SheetManager(set_code, cycle)
  players_new = sheet.GetPlayers()
  if cycle in (1,):
    sheet_old = sheet_manager.SheetManager(previous_set_code, 5)
    scores_old = {p.id: p.score for p in sheet_old.GetPlayers()}
    players = [
        p._replace(score=scores_old.get(p.id, fractions.Fraction(1, 2)))
        for p in players_new
    ]
  else:
    players = players_new
  # Sigma; Cycle 1: 0.1; 2: 0.05; 3, 4, 5: 0.0
  sigma = max(0, 0.15 - 0.05 * cycle)
  print(f'sigma = {sigma}')
  pairer = Pairer(players, sigma=sigma)
  pairer.GiveBye()
  start = time.time()
  pairings = pairer.MakePairings(random_pairings=False)
  pairings = OrderPairingsByTsp(pairings)
  if pairer.byed_player and any(pairing[0] == BYE for pairing in pairings):
    # If the BYE ended up in the left column, swap the columns.
    pairings = [(b, a) for (a, b) in pairings]
  if FLAGS.tabprint:
    for p, q in pairings:
      print(f'{p.name}\t{q.name}')
  else:
    PrintPairings(pairings)
  ValidatePairings(
      pairings, n=pairer.correct_num_matches + bool(pairer.byed_player))
  t = time.time() - start
  try:
    os.mkdir('pairings')
  except FileExistsError:
    pass
  with open(f'pairings/{set_code}{cycle}.{int(time.time())}.txt',
            'w') as output:
    PrintPairings(pairings, stream=output)
  with open(f'pairings/{set_code}{cycle}.txt', 'w') as output:
    PrintPairings(pairings, stream=output)
  print(f'Finished in {int(t // 60)}m{t % 60:.1f}s wall time.')

  if FLAGS.write:
    sheet.Writeback(pairings)


class Error(Exception):
  pass


class DuplicateMatchError(Error):
  """The same match-up appears twice in this set of pairings."""


class SelfMatchError(Error):
  """A player is match against themself."""


class RepeatMatchWarning(UserWarning):
  """A match-up from a previous round appears in this set of pairings."""


class WrongNumberOfMatchesError(Error):
  """This set of pairings has the wrong number of matches."""


if __name__ == '__main__':
  app.run(Main)
