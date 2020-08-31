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

from typing import List, Optional, Tuple
from absl import app
from absl import flags

import blitzstein_diaconis
import elkai
import magic_sets
import numpy as np
import player as player_lib
import sheet_manager

BYE = player_lib.Player('noreply', 'BYE', fractions.Fraction(0), 0, ())
EFFECTIVE_INFINITY = (1 << 31) - 1
FLAGS = flags.FLAGS
HUB_COST = 1
MAX_LCM = 10080  # 2 × 7!
MAX_PROCESSES = multiprocessing.cpu_count()

Pairings = List[Tuple[player_lib.Player, player_lib.Player]]

flags.DEFINE_bool('write',
                  False,
                  'Write the pairings to the spreadsheet',
                  short_name='w')
flags.DEFINE_bool(
    'fetch',
    False,
    'Force a fetch from the sheet, overriding the 20 minute cache timeout.',
    short_name='f')


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
  """Raises an error if the pairings aren't valid.

  Args:
    pairings: The proposed pairings.
    n: The expected number of pairings.

  Raises:
    WrongNumberOfMatchesError: There were not `n` matches.
    DuplicateMatchError: If the proposed pairings contain a duplicate.
    RepeatMatchError: If the proposed contain a match that occurred in a
    previous cycle.
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
    if p == q or p.id in q.opponents or q.id in p.opponents:
      raise RepeatMatchError(f'{p.id}, {q.id}')

def RoundTo(n, to):
  return int(n / to + 0.5) * to

def PrintPairings(pairings, stream=sys.stdout):
  """Print a pretty table of the model to the given stream."""
  with contextlib.redirect_stdout(stream):
    for (p, q) in pairings:
      # 6 + 6 + 28 + 28 + 4 spaces + "vs." (3) = 75
      p_score = f'{float(p.score):.3f}'.lstrip('0')
      q_score = f'{float(q.score):.3f}'.lstrip('0')
      n_stars = min(5, max(0, RoundTo(5 + math.log(max(0.00001, abs(p.score-q.score)), 2), 0.5)))
      star_string = '\u2b24' * int(n_stars)
      if n_stars % 1 == 0.5:  # Exact float comparison!? Should be OK because we just rounded to an exact power of 2.
        star_string += '\u25d6'
      if n_stars > 2:
        if stream.isatty():
          star_string= f'\033[1m{star_string}\033[0m'
      line = f'({p_score:>4}) {p.name:>28} vs. {q.name:<28} ({q_score:>4}) {star_string}'
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

  def __init__(self, players: List[player_lib.Player], sigma=0.0):
    self.sigma = sigma
    self.players = players
    self.players_by_id = {player.id: player for player in players}
    self.byed_player = None
    self.lcm = 1
    for d in set(p.score.denominator for p in self.players):
      self.lcm = Lcm(self.lcm, d)

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
      byed_player = min(eligible_players,
                        key=lambda p: (p.score, random.random()))
      self.players.remove(byed_player)
      self.byed_player = byed_player._replace(
          requested_matches=byed_player.requested_matches - 1)
      self.players.append(self.byed_player)
      return self.byed_player

  def MakePairings(self, random_pairings=False) -> Pairings:
    """Make pairings — random in cycle 1, else TSP optimized."""
    if random_pairings:
      print('Random pairings')
      pairings = self.RandomPairings()
    else:
      print('Optimizing pairings')
      pairings = self.TravellingSalesPairings()
    ValidatePairings(pairings, n=self.correct_num_matches)
    if self.byed_player:
      pairings.append((self.byed_player, BYE))
      ValidatePairings(pairings, n=self.correct_num_matches + 1)
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

  def TravellingSalesPairings(self):
    """Compute optimal pairings with a travelling-salesman solver."""
    odd_players = list(p for p in self.players if Odd(p.requested_matches))
    assert Even(len(odd_players))
    random.shuffle(odd_players)

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
            weights[i, j] = round(
                (my_lcm * (p.score - q.score + random.gauss(0, self.sigma)))**2)
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
      for s in TourSuccessors(tour, tsp_nodes):
        try:
          pri_q.put(s + (w,))
        except ValueError:
          pass
      semaphore.release()

    tour = elkai.solve_int_matrix(weights)
    for s in TourSuccessors(tour, tsp_nodes):
      pri_q.put(s + (weights,))
    spinner = itertools.cycle(['/', '—', '\\', '|'])
    with concurrent.futures.ProcessPoolExecutor(MAX_PROCESSES) as pool:
      while True:
        try:
          semaphore.acquire()
          num_dupes, edge_to_remove, pairings, weights = pri_q.get()
        except ValueError:
          continue
        print(f'\033[A\033[KEliminating {num_dupes} duplicate pairings... '
              f'{next(spinner)}')
        if not edge_to_remove:
          return pairings
        out, in_ = edge_to_remove
        weights = weights.copy()
        weights[out, in_] = EFFECTIVE_INFINITY
        weights[in_, out] = EFFECTIVE_INFINITY
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


def OrderPairingsByTsp(pairings: Pairings) -> Pairings:
  """Sort the given pairings by minimal cost tour."""
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
      weights[alpha_right,
              beta_left] = weights[alpha_left,
                                   beta_right] = (PairingTransitionCost(
                                       pairings[alpha], pairings[beta]))
      # normal;swapped
      # swapped;normal
      weights[alpha_right,
              beta_right] = weights[alpha_left,
                                    beta_left] = (PairingTransitionCost(
                                        pairings[alpha], pairings[beta][::-1]))
  tour = elkai.solve_float_matrix(weights)
  output_pairings = []
  for node in tour[1::2]:
    next_pairing = pairings[(node - 1) // 2]
    if node % 2 == 0:
      next_pairing = next_pairing[::-1]
    output_pairings.append(next_pairing)
  return output_pairings


def OrderPairingsByScore(pairings: Pairings) -> Pairings:
  return list(
      sorted(pairings, key=lambda t: (t[0].score, t[1].score, t), reverse=True))


def PairingTransitionCost(pairing_alpha, pairing_beta) -> float:
  left_cost = 1 - difflib.SequenceMatcher(a=pairing_alpha[0],
                                          b=pairing_beta[0]).ratio()
  right_cost = 1 - difflib.SequenceMatcher(a=pairing_alpha[1],
                                           b=pairing_beta[1]).ratio()
  return left_cost + right_cost


def Main(argv):
  """Fetch records from the spreadsheet, generate pairings, write them back."""
  set_code, cycle = argv[1:]
  cycle = int(cycle)
  previous_set_code = list(magic_sets.names.keys())[-2]
  sheet = sheet_manager.SheetManager(set_code, cycle)
  sheet_old = sheet_manager.SheetManager(previous_set_code, 5)
  players_new = sheet.GetPlayers()
  players_old = {p.id: p.score for p in sheet_old.GetPlayers()}
  players = [
      p._replace(score=(2 * p.score +
                        players_old.get(p.id, fractions.Fraction(1, 2))) / 3)
      for p in players_new
  ]
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
  PrintPairings(pairings)
  ValidatePairings(pairings,
                   n=pairer.correct_num_matches + bool(pairer.byed_player))
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


class RepeatMatchError(Error):
  """A match-up from a previous round appears in this set of pairings."""


class WrongNumberOfMatchesError(Error):
  """This set of pairings has the wrong number of matches."""


if __name__ == '__main__':
  app.run(Main)
