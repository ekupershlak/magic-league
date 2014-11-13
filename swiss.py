import copy
import collections
import pdb
import gspread
import itertools
import random
import sys
import xpermutations

sheets_account = 'chrisconnett@gmail.com'
sheets_password = 'fyiyjhehvjmrongv'
sheets_spreadsheet = 'magic-ny KTK Sealed League'
n = 3
limit = 40320

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
  total_mismatch = dict.fromkeys(names, 0)

  for i in range(1, n + 1):
    cycle = spreadsheet.worksheet('Cycle {}'.format(i))
    a = cycle.col_values(3)[1:]
    b = cycle.col_values(4)[1:]
    winners = cycle.col_values(7)[1:]
    if i >= 3:
      for pa, pb, winner in zip(a, b, winners):
        total_mismatch[pa] += partial_scores[pb] - partial_scores[pa]
        total_mismatch[pb] += partial_scores[pa] - partial_scores[pb]
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
  return groups, previous_pairings, total_mismatch

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
      total_mismatch[pa] += sb - sa
      total_mismatch[pb] += sa - sb
    acc.append(pairings)
  return sum(mismatch**2 for mismatch in total_mismatch.values()), acc


def main():
  groups, previous_pairings, total_mismatch = fetch()
  best = min(pair3(groups, previous_pairings, tm) for i in range(30))
  for round in best[0]:
    for (sa, pa), (sb, pb) in round:
        print '{}\t{}'.format(pa, pb)
