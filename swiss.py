"""Solver for swiss pairings."""

from __future__ import division

import collections
import cPickle
import datetime
import itertools
import fractions
import random
import time

import gspread
import z3
import password

sheets_account = 'chrisconnett@gmail.com'
sheets_password = password.sheets_password
sheets_spreadsheet = 'magic-ny DTK Sealed League'
cycle_to_pair = 3
num_cycles_previous = cycle_to_pair - 1
limit = 40320
BYE = 'BYE'


class NamedStack(z3.Solver):
  def __init__(self, *args, **kwargs):
    super(NamedStack, self).__init__(*args, **kwargs)
    self._names = {}
    self._depth = 0

  def push(self, name=None):
    if name:
      self._names[name] = self._depth
    super(NamedStack, self).push()
    self._depth += 1

  def pop(self, name=None):
    if name:
      while self._depth > self._names[name]:
        super(NamedStack, self).pop()
        self._depth -= 1
      self._names.pop(name)
    else:
      super(NamedStack, self).pop()


def Take(n, iterable):
  """Returns first n items of the iterable as a list."""
  return list(itertools.islice(iterable, n))


def Odd(n):
  return n % 2 == 1


def Even(n):
  return n % 2 == 0


def Lcm(a, b):
  return a * b / fractions.gcd(a, b)

def Timeleft(deadline):
  return int(deadline - time.time() + 0.5)


def GetSpreadsheet():
  session = gspread.login(sheets_account, sheets_password)
  return session.open(sheets_spreadsheet)


def Fetch():
  """Fetches data from the spreadsheet."""

  spreadsheet = GetSpreadsheet()
  standings = spreadsheet.worksheet('Standings')
  names = standings.col_values(2)[1:]
  wins, losses, draws = [
    [int(n) for n in standings.col_values(4 + c)[1:]]
    for c in range(3)]
  scores = [fractions.Fraction(3 * w, 3 * (w + l + d))
            for w, l, d in zip(wins, losses, draws)]
  lcm = reduce(Lcm, set(score.denominator for score in scores))
  print 'lcm is', lcm
  scores = [int(score * lcm) for score in scores]
  requested_matches = [int(s) for s in standings.col_values(
    9 + cycle_to_pair - 1)[1:]][::-1]

  previous_pairings = set()

  for i in range(1, num_cycles_previous + 1):
    cycle = spreadsheet.worksheet('Cycle {}'.format(i))
    a = cycle.col_values(2)[1:]
    b = cycle.col_values(3)[1:]
    winners = cycle.col_values(6)[1:]

    previous_pairings |= set(zip(a, b))
    previous_pairings |= set(zip(b, a))

  if Odd(sum(requested_matches)):
    targetted_for_bye = 3
    candidates = [
      (i, name) for i, (name, request) in enumerate(zip(names, requested_matches))
      if requested_matches[i] == targetted_for_bye and
      (name, BYE) not in previous_pairings]
    byed_i, byed_name = random.choice(candidates)
    requested_matches[byed_i] -= 1
    print byed_name, 'receives a bye.'

  groups = [list(group) for _, group in
            itertools.groupby(sorted(zip(scores, names), reverse=True),
                              key=lambda (s, n): s)]
  return groups, previous_pairings, requested_matches


def Writeback(pairings):
  spreadsheet = GetSpreadsheet()
  ws_name = 'Cycle ' + str(cycle_to_pair)
  output = spreadsheet.worksheet(ws_name)
  pairings_range = output.range('B2:C' + str(len(pairings) + 1))
  for cell, player in zip(
      pairings_range, (player for row in pairings for player in row)):
    cell.value = player
  print 'Writing to', ws_name
  output.update_cells(pairings_range)


def MakeSlots(s, n_players, r_rounds):
  """Creates output pairing variables."""
  slots = collections.defaultdict(dict)
  for n in range(n_players):
    for m in range(n_players):
      if n < m:
        slots[n][m] = z3.Bool('m_{},{}'.format(n, m))
    n_adjacency = []
    for m in range(n_players):
      if n < m:
        n_adjacency.append(slots[n][m])
      elif n > m:
        n_adjacency.append(slots[m][n])
    s.add(requested_matches[n] == z3.Sum([z3.If(slot, 1, 0) for slot in n_adjacency]))
  return slots

def RequestedMatches(s, slots, r):
  for n in range(n_players):
    n_adjacency = []
    for m in range(n_players):
      if n < m:
        n_adjacency.append(slots[n][m])
      elif n > m:
        n_adjacency.append(slots[m][n])
    if requested_matches[n] >= r:
      s.add(z3.Or(n_adjacency))
      s.add([z3.Implies(b, z3.Not(z3.Or([b2 for b2 in n_adjacency if b2 is not b])))
             for b in n_adjacency])
    else:
      s.add(z3.Not(z3.Or(n_adjacency)))


def NoRepeatMatches(s, slots, previous_pairings):
  for n, row in slots.items():
    for m, _ in row.items():
      if (reverse_players[n], reverse_players[m]) in previous_pairings:
        s.add(z3.Not(slots[n][m]))


def MismatchSum(slots, scores):
  terms = []
  sq_terms = []
  for n, row in slots.items():
    for m, slot in row.items():
      if n < m:
        terms.append(z3.If(slot, (scores[m] - scores[n]), 0))
        sq_terms.append(z3.If(slot, (scores[m] - scores[n]) ** 2, 0))
  return z3.Sum(terms), z3.Sum(sq_terms)

try:
  file('dat')
except IOError:
  cPickle.dump(Fetch(), file('dat', 'w'))
groups, previous_pairings, requested_matches = cPickle.load(file('dat'))

g_star = list(reversed(list(itertools.chain(*groups))))
players = {name: id for (id, (score, name)) in zip(itertools.count(), g_star)}
scores = {id: score for (id, (score, name)) in zip(itertools.count(), g_star)}

reverse_players = {number: name for name, number in players.items()}
player_scores = {reverse_players[id]: score for (id, score) in scores.items()}


def RemoveBye(l):
  return [p for p in l if p != BYE]

opponents = {}
for a, b in previous_pairings:
  if b != BYE:
    opponents.setdefault(a, []).append(b)
## omw = {
##     player: max(1 / 3.,
##                 sum(player_scores[opponent] -
##                     3 if BYE in opponents[player] else 0 /
##                     (3 * len(RemoveBye(opponents[player])))
##                     for opponent in opponents[player] if opponent != BYE) /
##                 len(RemoveBye(opponents[player])))
##     for player in players if player != BYE}


def Search(seconds=180, enumeration=None):
  """Constructs an SMT problem for pairings and solves it."""
  # import pdb; pdb.set_trace()
  s = z3.Solver()
  s.push()
  slots = MakeSlots(s, len(players), 3)
  NoRepeatMatches(s, slots, previous_pairings)
  all_metrics = []
  mismatch_sum_result = [MismatchSum(slots, scores)]
  for _, squared_mismatch in mismatch_sum_result:
    all_metrics.append(squared_mismatch)
    # all_metrics.append(linear_mismatch)
  metrics = all_metrics[:]

  deadline = time.time() + seconds
  metric = metrics.pop(0)
  while True:
    s.set('soft_timeout', Timeleft(deadline) * 1000)
    status = s.check()
    if status == z3.sat:
      model = s.model()
      badness = model.evaluate(metric)
      print 'Badness: {}'.format(tuple(model.evaluate(m) for m in all_metrics))
      s.push()
      if Timeleft(deadline) > 0:
        print 'Time left:', str(datetime.timedelta(seconds=Timeleft(deadline)))
        s.add(metric < badness)
      else:
        print 'Time limit reached.'
        s.add(metric == badness)
        break
    elif status == z3.unsat:
      print 'OPTIMAL!'
      print 'Badness: {}'.format(tuple(model.evaluate(m) for m in all_metrics))
      s.pop()
      s.push()
      s.add(metric == badness)
      try:
        metric = metrics.pop(0)
        s.push()
        badness = model.evaluate(metric)
        s.add(metric < badness)
      except IndexError:
        break
    else:
      print 'Time limit reached.'
      s.pop()
      s.push()
      s.add(metric <= badness)
      break

  winning_model = model
  if enumeration and status in (z3.unsat, z3.unknown):
    total = 0
    for i, m in enumerate(AllOptimalModels(s, slots, deadline + enumeration)):
      print i
      total += 1
      if random.random() < 1.0 / total:
        winning_model = m
    print 'Total solutions found:', max(total, 1)

  PrintModel(slots, scores, winning_model)
  print
  print 'Badness:', tuple(winning_model.evaluate(m) for m in all_metrics)
  return list(ModelPlayers(slots, winning_model))


def NegateModel(slots, model):
  return z3.Or([slot != model[slot]
                for d in slots.values()
                for slot in d.values()])


def AllOptimalModels(s, slots, deadline=None):
  try:
    s.push()
    while True:
      if deadline:
        if Timeleft(deadline) > 0:
          s.set('soft_timeout', Timeleft(deadline) * 1000)
        else:
          return
      if s.check() == z3.sat:
        model = s.model()
        yield model
        s.add(NegateModel(slots, model))
      else:
        break
  finally:
    s.pop()


def PrintModel(slots, scores, model):
  for n, row in reversed(slots.items()):
    for m, playing in reversed(row.items()):
      if str(model.evaluate(playing)) == 'True':
        player = reverse_players[m]
        opponent = reverse_players[n]
        print '{:>6} {:>20} vs. {:<20} {:>6}'.format(
            '({})'.format(scores[m]), player, opponent,
            '({})'.format(scores[n]))


def ModelPlayers(slots, model):
  for n, row in reversed(slots.items()):
    for m, playing in reversed(row.items()):
      if str(model.evaluate(playing)) == 'True':
        yield (reverse_players[m], reverse_players[n])
