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

def make_slots(env, n_players, r_rounds):
  """Creates output pairing variables."""
  int_type = msat_get_integer_type(env)
  zero = msat_make_number(env, '0')
  upper_bound = msat_make_number(env, str(n_players - 1))
  slots = []
  for r in range(r_rounds):
    round_slots = []
    slots.append(round_slots)
    for n in range(n_players):
      slot = msat_declare_function(env, 'r_{}-{}'.format(r, n), int_type)
      # Bounded
      msat_assert_formula(env, msat_make_leq(env, zero, slot))
      msat_assert_formula(env, msat_make_leq(env, slot, upper_bound))
      round_slots.append(slot)
      # Sorted
      if n != 0:
        msat_assert_formula(
          env, msat_make_leq(env, round_slots[n], round_slots[n - 1]))
    # Distinct
    msat_assert_formula(
      env, msat_from_string(env, '(distinct {})'.format(
        ' '.join('r_{}-{}'.format(r, n) for n in range(n_players)))))

  return slots

def make_score_function(env, scores):
  """Creates match points score mapping function."""
  int_type = msat_get_integer_type(env)
  score_function_type = msat_get_function_type(env, [int_type], int_type)
  func = msat_declare_function(env, 'score', score_function_type)
  for player_id, score in scores.items():
    msat_assert_formula(msat_make_equal(
      env, msat_make_number(env, str(player_id)),
      msat_make_number(env, str(score))))
  return func

def make_played_function(env, previous_pairings):
  int_type = msat_get_integer_type(env)
  bool_type = msat_get_bool_type(env)
  played_function_type = msat_get_function_type(
    env, [int_type, int_type], bool_type)
  true = msat_make_true(env)
  id_mapping = player_id_mapping(previous_pairings)

  func = msat_declare_function(env, 'played', pair_function_type)
  for (pa, pb) in previous_pairings:
    msat_assert_formula(
      env, msat_assert_equal(env, true, msat_make_uf(
        env, func, [msat_make_number(env, str(id_mapping[pa])),
                    msat_make_number(env, str(id_mapping[pb]))])))
    msat_assert_formula(
      env, msat_assert_equal(env, true, msat_make_uf(
        env, func, [msat_make_number(env, str(id_mapping[pb])),
                    msat_make_number(env, str(id_mapping[pa]))])))
  return func

def no_repeat_matches(env, slots, pair_func):
  false = msat_make_false(env)
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      if odd(n):
        msat_assert_formula(
          env, msat_make_equal(env, false, msat_make_uf(
            env, pair_func, [slots[r][n-1], slots[r][n]])))
    if odd(r) and odd(len(slots[0])):
      msat_assert_formula(
        env, msat_make_equal(env, false, msat_make_uf(
          env, pair_func, [slots[r-1][-1], slots[r][-1]])))

def odd(n):
  return n % 2 == 1

def no_repeat_byes(env, slots, previous_pairings):
  previously_byed = [player for (player, bye) in previous_pairings
                     if bye == BYE]
  id_mapping = player_id_mapping(previous_pairings)
  for player in previously_byed:
    msat_assert_formula(env, msat_make_not(
      env, msat_make_equal(env, msat_make_number(
        env, str(id_mapping[player])), slots[-1][-1])))

def player_id_mapping(previous_pairings):
  all_names = (set(pair[0] for pair in previous_pairings) +
               set(pair[1] for pair in previous_pairings))
  all_names -= set([BYE])
  return dict(zip(sorted(all_names), itertools.count()))

def metric1(env):
  for player, number in player_id_mapping.items():


def main():
  groups, previous_pairings, total_mismatch, player_mismatch = fetch()
  id_mapping = player_id_mapping(previous_pairings)
  scores = {id_mapping[player_name]: score
            for player_name, score in itertools.chain(*groups)}
  best = min(pair3(groups, previous_pairings, tm) for i in range(30))
  for round in best[0]:
    for (sa, pa), (sb, pb) in round:
        print '{}\t{}'.format(pa, pb)
