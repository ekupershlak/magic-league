import copy
import collections
import pdb
import gspread
import itertools
import random
import sys
import xpermutations

from mathsat import *
import z3

sheets_account = 'chrisconnett@gmail.com'
sheets_password = 'fyiyjhehvjmrongv'
sheets_spreadsheet = 'magic-ny KTK Sealed League'
n = 3
limit = 40320
BYE = 'BYE'

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def odd(n):
  return n % 2 == 1

def even(n):
  return n % 2 == 0

class CouldNotPairError(Exception):
  """Could not pair."""

def fetch():
  session = gspread.login(sheets_account, sheets_password)
  spreadsheet = session.open(sheets_spreadsheet)

  standings = spreadsheet.worksheet('Standings')
  names = standings.col_values(2)[1:]
  scores = [int(s) for s in standings.col_values(4)[1:]]

  previous_pairings = set()
  partial_scores = dict.fromkeys(names, 0)
  total_mismatch = 0
  player_mismatch = dict.fromkeys(names, 0)

  for i in range(1, n + 1):
    cycle = spreadsheet.worksheet('Cycle {}'.format(i))
    a = cycle.col_values(3)[1:]
    b = cycle.col_values(4)[1:]
    winners = cycle.col_values(7)[1:]
    for pa, pb, winner in zip(a, b, winners):
      player_mismatch[pa] += partial_scores[pb] - partial_scores[pa]
      player_mismatch[pb] += partial_scores[pa] - partial_scores[pb]
      total_mismatch += (partial_scores[pa] - partial_scores[pb]) ** 2

    for pa, pb, winner in zip(a, b, winners):
      partial_scores[pa] += {
        pa: 3,
        "Didn't play:{}-{}".format(pa, pb): 1,
        pb: 0}[winner]
      partial_scores[pb] += {
        pb: 3,
        "Didn't play:{}-{}".format(pa, pb): 1,
        pa: 0}[winner]
    previous_pairings |= set(zip(a, b))
    previous_pairings |= set(zip(b, a))

    groups = [list(group) for score, group in
              itertools.groupby(sorted(zip(scores, names), reverse=True),
                                key=lambda (s, n): s)]
  return groups, previous_pairings, total_mismatch, player_mismatch

def pop_pairup(groups):
  for group in groups:
    if group:
      pairup = random.choice(group)
      group.remove(pairup)
      return pairup
  raise CouldNotPairError()

def pair(groups, previous_pairings):
  if not groups:
    return []
  groups = collections.deque(copy.deepcopy(groups))
  pairings = []
  current = groups.popleft()
  while True:
    if len(current) % 2 == 1:
      pairup = pop_pairup(groups)
      current.append(pairup)
    random.shuffle(current)
    if len(current) > 16:
      raise CouldNotPairError()
    for permutation in take(limit, xpermutations.xpermutations(current)):
      group_pairings = []
      while permutation:
        pa = permutation[0]
        pb = permutation[1]
        if (pa[1], pb[1]) not in previous_pairings:
          group_pairings.append((pa, pb))
          permutation = permutation[2:]
        else:
          break
      if len(group_pairings) == len(current) / 2:
        try:
          pairings.extend(group_pairings + pair(groups, previous_pairings))
          return pairings
        except CouldNotPairError:
          pass
    if groups:
      super_pairup = pop_pairup(groups)
      current.append(super_pairup)
    else:
      raise CouldNotPairError('Could not pair')

def pair3(groups, previous_pairings, total_mismatch):
  previous_pairings = copy.deepcopy(previous_pairings)
  total_mismatch = copy.deepcopy(total_mismatch)
  acc = []
  for i in range(1, 4):
    pairings = pair(groups, previous_pairings)
    for (sa, pa), (sb, pb) in pairings:
      previous_pairings.add((pa, pb))
      previous_pairings.add((pb, pa))
      total_mismatch[pa] += (sb - sa) ** 2
      total_mismatch[pb] += (sa - sb) ** 2

    acc.append(pairings)
  return sum(mismatch**2 for mismatch in total_mismatch.values()), acc

def MakeSlots(s, n_players, r_rounds):
  """Creates output pairing variables."""
  slots = []
  for r in range(r_rounds):
    round_slots = []
    slots.append(round_slots)
    for n in range(n_players):
      slot = z3.Int('r_{},{}'.format(r, n))
      round_slots.append(slot)
      # Bounded
      s.add(0 <= slot)
      s.add(slot < n_players)
    # Distinct
    s.add(z3.Distinct(*round_slots))
  return slots

def MakeScoreFunction(s, scores):
  """Creates match points score mapping function."""
  f = z3.Function('score', z3.IntSort(), z3.IntSort())
  for player_id, score in scores.items():
    s.add(f(player_id) == score)
  return f

def SortSlotsByScore(s, slots, score):
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      # Sorted
      if odd(n):
        # Right-column <= left-column opponent.
        s.add(round_slots[n] <= round_slots[n - 1])
      if n not in (0, 1) and not (even(n) and n == len(round_slots) - 1):
        # Each player <= player above in same column.
        s.add(round_slots[n] <= round_slots[n - 2])


def MakePlayedFunction(s, previous_pairings, players):
  played = z3.Function('played', z3.IntSort(), z3.IntSort(), z3.BoolSort())

  for (pa, pb) in previous_pairings:
    s.add(played(players[pa], players[pb]))
  for id_a in players.values():
    s.add(played(id_a, id_a))
  return played

def NoRepeatMatches(s, slots, played):
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      if odd(n):
        s.add(z3.Not(played(slots[r][n-1], slots[r][n])))
    if odd(r) and odd(len(slots[0])):
      s.add(z3.Not(played(slots[r-1][-1], slots[r][-1])))

def NoRepeatByes(s, slots, previous_pairings, players):
  previously_byed = [player for (player, bye) in previous_pairings
                     if bye == BYE]
  for player in previously_byed:
    s.add(slots[-1][-1] != players[player])

# Metric 1
def PerPlayerAbsoluteMismatchSumSquared(s, slots, players, score_func):
  mismatches = [
    z3.Function('abs_mismatch_' + str(r), z3.IntSort(), z3.IntSort())
    for r in range(len(slots))]
  for round_slots, round_mismatch in zip(slots, mismatches):
    for n, slot in enumerate(round_slots):
      if odd(n):
        player = slot
        opponent = round_slots[n - 1]
        s.add(round_mismatch(opponent) == round_mismatch(player),
              round_mismatch(player) ==
              score_func(opponent) - score_func(player))
    if odd(len(round_slots)):
      s.add(round_mismatch[-1] == score_func(round_slots[-1]))

  def AbsoluteMismatchSum(player):
    return z3.Sum(*[round_mismatch(player) for round_mismatch in mismatches])

  return z3.Sum(*[AbsoluteMismatchSum(player_id) ** 2
                  for player_id in players.values()])

def PerPlayerSquaredSumMismatch(s, slots, players, score_func):
  pass

def main():
  ## best = min(pair3(groups, previous_pairings, tm) for i in range(30))
  ## for round in best[0]:
  ##   for (sa, pa), (sb, pb) in round:
  ##       print '{}\t{}'.format(pa, pb)
  pass

import cPickle
try:
  file('dat')
except IOError:
  cPickle.dump(fetch(), file('dat', 'w'))
groups, previous_pairings, total_mismatch, player_mismatch = cPickle.load(
  file('dat'))


scores = {id: score for (id, (score, name)) in
          zip(itertools.count(), reversed(list(itertools.chain(*groups))))}
players = {name: id for (id, (score, name)) in
           zip(itertools.count(), reversed(list(itertools.chain(*groups))))}

s = z3.Solver()
slots = MakeSlots(s, len(players), 1)
score = MakeScoreFunction(s, scores)
SortSlotsByScore(s, slots, score)
played = MakePlayedFunction(s, previous_pairings, players)
NoRepeatMatches(s, slots, played)
NoRepeatByes(s, slots, previous_pairings, players)
metric1 = PerPlayerAbsoluteMismatchSumSquared(s, slots, players, score)

def PrintModel(slots, players, score, model):
  reverse_players = {number: name for name, number in players.items()}
  for r, round_slots in enumerate(slots):
    print 'Round', r + 1
    for n, slot in enumerate(round_slots):
      if odd(n):
        player = slot
        opponent = round_slots[n - 1]
        print '{:>4} {:>20} vs. {:<20} {:>4}'.format(
          '({})'.format(model.evaluate(score(opponent))),
          reverse_players[model.evaluate(opponent).as_long()],
          reverse_players[model.evaluate(player).as_long()],
          '({})'.format(model.evaluate(score(player))))
  print
  print 'Badness:', model.evaluate(metric1)
