# -*- encoding: utf-8 -*- python3
"""Solver for swiss pairings."""

import argparse
import collections
import concurrent.futures
import contextlib
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

BYE = player_lib.Player('noreply', 'BYE', fractions.Fraction(0), 0, ())
EFFECTIVE_INFINITY = (1 << 31) - 1
FLAGS = None  # Parsing the flags needs to happen in main.
HUB_COST = 1
MAX_PROCESSES = multiprocessing.cpu_count()
MAX_LCM = 10080  # 2 × 7!
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


def ValidatePairings(pairings: Pairings, n: Optional[int] = None) -> None:
  """Raises an error if the pairings aren't valid."""
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
    if p == q or p.id in q.opponents or q.id in p.opponents:
      raise RepeatMatchError(f'{p.id}, {q.id}')


def SplitOnce(pairings: Pairings) -> Tuple[Pairings, Pairings]:
  """Split the pairings loop into two loops at the best crossover point."""
  best_split = (pairings, [])
  best_loss = SSE(pairings)
  for i in range(len(pairings)):
    for j in range(i, len(pairings)):
      left = pairings[j + 1:] + pairings[:i]
      right = pairings[i + 1:j]
      left.append((pairings[i][0], pairings[j][1]))
      right.append((pairings[j][0], pairings[i][1]))
      try:
        ValidatePairings(left + right)
      except Error:
        continue
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
    approx_loss = loss.limit_denominator(1000)
    approx_string = 'Approx. ' if approx_loss != loss else ''
    print(f'Sum of squared error: {approx_string}{approx_loss!s}')
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

  @property
  def correct_num_matches(self):
    """Returns the number of non-BYE matches that there *should* be."""
    return sum(player.requested_matches for player in self.players) // 2

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
      print('Optimizing pairings')
      pairings = self.TravellingSalesPairings()
      ValidatePairings(pairings, n=self.correct_num_matches)
      print('Searching for final augmenting swaps.')
      pairings = SplitAll(pairings)
    ValidatePairings(pairings, n=self.correct_num_matches)
    if self.bye:
      pairings.append((self.bye, BYE))
      ValidatePairings(pairings, n=self.correct_num_matches + 1)
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
    random.shuffle(odd_players)
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
    my_lcm = min(MAX_LCM, self.lcm)
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
            weights[i, j] = (int(p.score * my_lcm) - int(q.score * my_lcm))**2
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
          weights[i, j] = (HUB_COST * my_lcm)**2
        else:
          assert False, f'{p.id} {ptype} -- {q.id} {qtype}'

    pairings = []
    pri_q = queue.PriorityQueue()
    semaphore = threading.BoundedSemaphore(MAX_PROCESSES)

    def AfterSolve(future):
      w, tour = future.result()
      semaphore.release()
      for s in TourSuccessors(tour, tsp_nodes):
        try:
          pri_q.put(s + (w,))
        except ValueError:
          pass

    tour = elkai.solve_int_matrix(weights)
    for s in TourSuccessors(tour, tsp_nodes):
      pri_q.put(s + (weights,))
    spinner = itertools.cycle(['/', '—', '\\', '|'])
    with concurrent.futures.ProcessPoolExecutor(MAX_PROCESSES) as pool:
      while True:
        try:
          num_dupes, edge_to_remove, pairings, weights = pri_q.get()
        except ValueError:
          continue
        print(f'\033[A\033[KEliminating {num_dupes} duplicate pairings... '
              f'{next(spinner)}')
        if not edge_to_remove:
          pool.shutdown(wait=False)
          return pairings
        out, in_ = edge_to_remove
        weights = weights.copy()
        weights[out, in_] = EFFECTIVE_INFINITY
        weights[in_, out] = EFFECTIVE_INFINITY
        semaphore.acquire()
        future = pool.submit(SolveWeights, weights)
        future.add_done_callback(AfterSolve)


def TourSuccessors(tour: List[int], tsp_nodes):
  """Yield pairings and nominate one dupe-match edge to be removed."""
  pairings = []
  edges_to_remove = []
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
        edges_to_remove.append((out, in_))
      else:
        pairings.append((p, q))
  if not edges_to_remove:
    yield (0, None, pairings)
  else:
    for edge in edges_to_remove:
      yield (len(edges_to_remove), edge, pairings)


def SolveWeights(weights):
  return weights, elkai.solve_int_matrix(weights)


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
  with open(
      f'pairings/pairings-{FLAGS.set_code}{FLAGS.cycle}.{int(time.time())}.txt',
      'w') as output:
    PrintPairings(pairings, stream=output)
  with open(f'pairings/pairings-{FLAGS.set_code}{FLAGS.cycle}.txt',
            'w') as output:
    PrintPairings(pairings, stream=output)

  if FLAGS.write:
    sheet.Writeback(sorted(pairings))


class Error(Exception):
  pass


class DuplicateMatchError(Error):
  """The same match-up appears twice in this set of pairings."""


class RepeatMatchError(Error):
  """A match-up from a previous round appears in this set of pairings."""


class WrongNumberOfMatchesError(Error):
  """This set of pairings has the wrong number of matches."""


if __name__ == '__main__':
  FLAGS = flags.parse_args(sys.argv[1:])
  Main()
