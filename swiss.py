import collections
import copy
import datetime
import gspread
import itertools
import password
import random
import sys
import time
import xpermutations
import z3

sheets_account = 'chrisconnett@gmail.com'
sheets_password = password.sheets_password
sheets_spreadsheet = 'magic-ny KTK Sealed League'
cycle_to_pair = 5
n = cycle_to_pair - 1
limit = 40320
BYE = 'BYE'

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def odd(n):
  return n % 2 == 1

def even(n):
  return n % 2 == 0

def timeleft(deadline):
  return int(deadline - time.time() + 0.5)

class CouldNotPairError(Exception):
  """Could not pair."""

def GetSpreadsheet():
  session = gspread.login(sheets_account, sheets_password)
  return session.open(sheets_spreadsheet)


def Fetch():
  spreadsheet = GetSpreadsheet()
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
        '': 1,
        pb: 0}[winner]
      partial_scores[pb] += {
        pb: 3,
        "Didn't play:{}-{}".format(pa, pb): 1,
        '': 1,
        pa: 0}[winner]
    previous_pairings |= set(zip(a, b))
    previous_pairings |= set(zip(b, a))

    groups = [list(group) for score, group in
              itertools.groupby(sorted(zip(scores, names), reverse=True),
                                key=lambda (s, n): s)]
  return groups, previous_pairings, total_mismatch, player_mismatch

def Writeback(pairings):
  spreadsheet = GetSpreadsheet()
  ws_name = 'Cycle ' + str(cycle_to_pair)
  output = spreadsheet.worksheet(ws_name)
  pairings_range = output.range('C2:D' + str(len(pairings) + 1))
  for cell, player in zip(
      pairings_range, (player for row in pairings for player in row)):
    cell.value = player
  print 'Writing to', ws_name
  output.update_cells(pairings_range)


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

def PermutationPair():
  best = min(pair3(groups, previous_pairings, tm) for i in range(30))
  for round in best[0]:
    for (sa, pa), (sb, pb) in round:
        print '{}\t{}'.format(pa, pb)

# SMT Based Solver below

def MakeSlots(s, n_players, r_rounds):
  """Creates output pairing variables."""
  slots = []
  for r in range(r_rounds):
    round_slots = {}
    slots.append(round_slots)
    for n in range(n_players):
      round_slots[n] = {}
      for m in range(n_players):
        if n < m:
          round_slots[n][m] = z3.Bool('r_{},{},{}'.format(r, n, m))
      n_adjacency = []
      for m in range(n_players):
        if n < m:
          n_adjacency.append(round_slots[n][m])
        elif n > m:
          n_adjacency.append(round_slots[m][n])
      for p in n_adjacency:
        # At most one opponent
        opps = [q for q in n_adjacency if p is not q]
        s.add(z3.Implies(p, z3.Not(z3.Or(opps))))
      # At least one opponent
      s.add(z3.Or(n_adjacency))
  return slots

def MakePlayedFunction(s, slots, previous_pairings, players):
  played_0 = {}
  for n, row in slots[0].items():
    for m, _ in row.items():
      played_0.setdefault(
        n, {}).setdefault(m, z3.Bool('played_0,{},{}'.format(n, m)))
      # Previous cycles' pairings
      if (n, m) in previous_pairings:
        s.add(played_0[n][m])
      else:
        s.add(z3.Not(played_0[n][m]))

  played_funcs = [played_0]
  for r, round_slots in enumerate(slots):
    if r == 0:
      continue
    # Matches from earlier rounds count as played for later rounds.
    played_prime = {}
    z3.Function('played_' + str(r),
                               z3.IntSort(), z3.IntSort(), z3.BoolSort())
    for n, row in slots[0].items():
      for m, _ in row.items():
        played_prime.setdefault(
          n, {}).setdefault(m, z3.Bool('played_0,{},{}'.format(n, m)))
    s.add([z3.Implies(played_funcs[-1][n][m], played_prime[n][m])
           for n in range(len(round_slots))
           for m in range(len(round_slots)) if n < m])
    ## for n, slot in enumerate(round_slots):
    ##   if odd(n):
    ##     # Set last round's matches as played
    ##     s.add(played_prime(slots[r-1][n-1], slots[r-1][n]))
    # TODO: If pairing more than 3 rounds, keep adding cross-round odd matches.
    played_funcs.append(played_prime)

  return played_funcs

def NoRepeatMatches(s, slots, played_funcs):
  for r, round_slots in enumerate(slots):
    played = played_funcs[r]
    for n, row in round_slots.items():
      for m, _ in row.items():
        if n < m:
          s.add(z3.Implies(played[n][m], z3.Not(round_slots[n][m])))

def NoRepeatByes(s, slots, previous_pairings, players):
  previously_byed = [player for (player, bye) in previous_pairings
                     if bye == BYE]
  for player in previously_byed:
    s.add(slots[-1][-1] != players[player])

def SignedMismatch(round_slots, scores, n, m):
  return z3.If(round_slots[n][m], scores[m] - scores[n], 0)

# Metric 1
def PerPlayerAbsoluteMismatchSumSquared(s, slots, players, score_func):
  mismatches = {}
  for r, round_slots in enumerate(slots):
    for n, row in round_slots.items():
      for m, slot in row.items():
        term = z3.If(slot, scores[m] - scores[n], 0)
        mismatches[n].append(term)
        mismatches[m].append(term)
  def PlayersMismatchSumSquared():
    for player_id in players.values():
      term_sum = z3.Sum(mismatches[player_id])
      yield term_sum * term_sum
  return z3.Sum(list(PlayersMismatchSumSquared()))

def MismatchSum(s, slots, scores):
  terms = []
  for r, round_slots in enumerate(slots):
    for n, row in round_slots.items():
      for m, slot in row.items():
        terms.append(z3.If(slot, scores[m] - scores[n], 0))
    yield z3.Sum(terms), z3.Sum([t * t for t in terms])

def MaximumMismatch(s, slots, score):
  maximum = z3.Int('maximum')
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      if odd(n):
        s.add(score(round_slots[n - 1]) - score(round_slots[n]) <= maximum)
  return maximum

def PerPlayerSquaredSumMismatch(s, slots, players, score_func):
  mismatches = [
    z3.Function('signed_mismatch_' + str(r), z3.IntSort(), z3.IntSort())
    for r in range(len(slots))]
  for round_slots, round_mismatch in zip(slots, mismatches):
    for n, slot in enumerate(round_slots):
      if odd(n):
        player = slot
        opponent = round_slots[n - 1]
        s.add(round_mismatch(opponent) == -round_mismatch(player),
              round_mismatch(player) ==
              score_func(opponent) - score_func(player))
      # TODO: odd players in a round
  def SignedMismatchSum(player):
    return z3.Sum(*[round_mismatch(player) for round_mismatch in mismatches])

  return z3.Sum([SignedMismatchSum(player_id) *
                 SignedMismatchSum(player_id)
                 for player_id in players.values()])

import cPickle
try:
  file('dat')
except IOError:
  cPickle.dump(Fetch(), file('dat', 'w'))
groups, previous_pairings, total_mismatch, player_mismatch = cPickle.load(
  file('dat'))


scores = {id: score for (id, (score, name)) in
          zip(itertools.count(), reversed(list(itertools.chain(*groups))))}
players = {name: id for (id, (score, name)) in
           zip(itertools.count(), reversed(list(itertools.chain(*groups))))}
reverse_players = {number: name for name, number in players.items()}

def Search(seconds=180, enumeration=None):
  s = z3.Solver()
  s.push()
  slots = MakeSlots(s, len(players), 1)
  played = MakePlayedFunction(s, slots, previous_pairings, players)
  NoRepeatMatches(s, slots, played)
  #NoRepeatByes(s, slots, previous_pairings, players)
  all_metrics = []
  mismatch_sum_result = list(MismatchSum(s, slots, scores))
  for linear_mismatch, _ in mismatch_sum_result:
    all_metrics.append(linear_mismatch)
  #for _, squared_mismatch in mismatch_sum_result:
  #  all_metrics.append(squared_mismatch)
  #all_metrics.append(
  #  PerPlayerAbsoluteMismatchSumSquared(s, slots, players, score))
  #all_metrics.append(PerPlayerSquaredSumMismatch(s, slots, players, score))
  metrics = all_metrics[:]

  deadline = time.time() + seconds
  metric = metrics[0]
  while True:
    s.set('soft_timeout', timeleft(deadline) * 1000)
    status = s.check()
    if status == z3.sat:
      model = s.model()
      badness = model.evaluate(metric)
      print 'Badness: {}'.format(tuple(model.evaluate(m) for m in all_metrics))
      s.push()
      if timeleft(deadline) > 0:
        print 'Time left:', str(datetime.timedelta(seconds=timeleft(deadline)))
        s.add(metric < badness)
        metric = max(metrics, key=lambda m: model.evaluate(m).as_long())
      else:
        print 'Time limit reached.'
        s.add(metric == badness)
        break
    elif status == z3.unsat:
      print 'OPTIMAL!'
      s.pop()
      s.add(metric == badness)
      try:
        metric = metrics.pop(0)
        s.add(metric < model.evaluate(metric))
      except IndexError:
        break
    else:
      print 'Time limit reached.'
      s.pop()
      s.add(metric <= badness)
      break

  winner = model
  if enumeration and status in (z3.unsat, z3.unknown):
    total = 0
    for i, m in enumerate(AllOptimalModels(s, slots, deadline + enumeration)):
      print i
      total += 1
      if random.random() < 1.0 / total:
        winner = m
    print 'Total solutions found:', max(total, 1)

  PrintModel(slots, players, scores, winner)
  print
  print 'Badness:', tuple(winner.evaluate(m) for m in all_metrics)
  return list(ModelPlayers(slots, players, score, winner))

def NegateModel(slots, model):
  return z3.Or([slot != model[slot] for round in slots for slot in round])

def AllOptimalModels(s, slots, deadline=None):
  while True:
    if deadline:
      if timeleft(deadline) > 0:
        s.set('soft_timeout', timeleft(deadline) * 1000)
      else:
        return
    if s.check() == z3.sat:
      model = s.model()
      yield model
      s.add(NegateModel(slots, model))
    else:
      break

def PrintModel(slots, players, scores, model):
  for r, round_slots in enumerate(slots):
    print 'Round', r + 1
    for n, slot in enumerate(round_slots):
      if odd(n):
        player = slot
        opponent = round_slots[n - 1]
        print '{:>4} {:>20} vs. {:<20} {:>4}'.format(
          '({})'.format(model.evaluate(scores[opponent])),
          reverse_players[model.evaluate(opponent).as_long()],
          reverse_players[model.evaluate(player).as_long()],
          '({})'.format(model.evaluate(scores[player])))

def ModelPlayers(slots, players, score, model):
  reverse_players = {number: name for name, number in players.items()}
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      if odd(n):
        player = slot
        opponent = round_slots[n - 1]
        yield (reverse_players[model.evaluate(opponent).as_long()],
               reverse_players[model.evaluate(player).as_long()])
