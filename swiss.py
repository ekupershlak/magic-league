"""Solver for swiss pairings."""

from __future__ import division

import collections
import cPickle
import datetime
import fractions
import itertools
import random
import time

import z3
import password

sheets_spreadsheet = 'magic-ny DTK Sealed League'
cycle_to_pair = 4
num_cycles_previous = cycle_to_pair - 1
BYE = 'BYE'


class NamedStack(z3.Solver):

  def __init__(self, *args, **kwargs):
    z3.Solver.__init__(self, *args, **kwargs)
    self._names = {}
    self._depth = 0

  def push(self, name=None):
    if name:
      self._names[name] = self._depth
    z3.Solver.push(self)
    self._depth += 1

  def pop(self, name=None):
    if name:
      while self._depth > self._names[name]:
        z3.Solver.pop(self)
        self._depth -= 1
      self._names.pop(name)
    else:
      z3.Solver.pop(self)
      self._depth -= 1


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
  return password.gc.open(sheets_spreadsheet)


def Fetch():
  """Fetches data from the spreadsheet."""

  spreadsheet = GetSpreadsheet()
  standings = spreadsheet.worksheet('Standings')
  names = standings.col_values(2)[1:]
  wins, losses, draws = [
      [int(n) for n in standings.col_values(4 + c)[1:]] for c in range(3)
  ]
  scores = [fractions.Fraction(3 * w, 3 * (w + l + d)) if w + l + d else
            fractions.Fraction(1, 2) for w, l, d in zip(wins, losses, draws)]
  lcm = 1
  for d in set(score.denominator for score in scores):
    lcm = Lcm(lcm, d)
  print 'lcm is', lcm

  requested_matches = [int(s)
                       for s in standings.col_values(9 + cycle_to_pair - 1)[1:]]
  names, requested_matches = zip(*[(n, rm)
                                   for (n, rm) in zip(names, requested_matches)
                                   if 0 < rm <= 3])
  names = list(names)
  requested_matches = list(requested_matches)
  print zip(names, requested_matches)
  print requested_matches
  print sum(requested_matches)

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
        (i, name)
        for i, (name, request) in enumerate(zip(names, requested_matches))
        if requested_matches[i] == targetted_for_bye and (
            name, BYE) not in previous_pairings
    ]
    byed_i, byed_name = random.choice(candidates)
    requested_matches[byed_i] -= 1
    print byed_name, 'receives a bye.'

  return zip(names, scores), previous_pairings, requested_matches


def Writeback(pairings):
  spreadsheet = GetSpreadsheet()
  ws_name = 'Cycle ' + str(cycle_to_pair)
  output = spreadsheet.worksheet(ws_name)
  pairings_range = output.range('B2:C' + str(len(pairings) + 1))
  for cell, player in zip(pairings_range,
                          (player for row in pairings for player in row)):
    cell.value = player
  print 'Writing to', ws_name
  output.update_cells(pairings_range)


def MakeSlots(n_players, r_rounds):
  """Creates output pairing variables."""
  slots = collections.defaultdict(dict)
  for n in range(n_players):
    for m in range(n_players):
      if n < m:
        slots[n][m] = z3.Bool('m_{},{}'.format(n, m))
  return slots


def ExactlyOne(vs):
  at_least_one = z3.Or(vs)
  at_most_one = z3.And([z3.Implies(v, z3.Not(z3.Or([w for w in vs if w is not v
                                                   ]))) for v in vs])
  return z3.And(at_least_one, at_most_one)


def PopCount(vs, n):
  if n == 0:
    return z3.Not(z3.Or(vs))
  else:
    terms = [z3.Or(vs)]
    for i, v in enumerate(vs):
      before = vs[:i]
      after = vs[i + 1:]
      terms.append(z3.Or([z3.And(
          z3.Implies(v, PopCount(before, a)), z3.Implies(v, PopCount(
              after, n - 1 - a))) for a in range(n)]))
    return z3.And(terms)


def EnumeratedPopCount(vs, n):
  if n == 0:
    return z3.Not(z3.Or(vs))
  else:
    options = []
    for combo in itertools.combinations(enumerate(vs), n):
      options.append(z3.And([v if (i, v) in combo else z3.Not(v)
                             for (i, v) in enumerate(vs)]))
    return z3.Or(options)


def RequestedMatches(slots, guaranteed, requested_matches):
  n_players = len(slots) + 1

  for n in range(n_players):
    print reverse_players[n], 'requests', requested_matches[n], 'matches'
    n_adjacency = []
    for m in range(n_players):
      if (n, m) not in guaranteed and (m, n) not in guaranteed:
        if n < m:
          n_adjacency.append(slots[n][m])
        elif n > m:
          n_adjacency.append(slots[m][n])
    # yield PopCount(n_adjacency, requested_matches[n])
    yield EnumeratedPopCount(n_adjacency, requested_matches[n])
  for i, _ in enumerate(requested_matches):
    requested_matches[i] = 0


def RequestedMatchesAll(slots):
  n_players = len(slots) + 1
  for n in range(n_players):
    n_adjacency = []
    for m in range(n_players):
      if n < m:
        n_adjacency.append(slots[n][m])
      elif n > m:
        n_adjacency.append(slots[m][n])
    yield requested_matches[n] == z3.Sum([z3.If(match, 1, 0)
                                          for match in n_adjacency])


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
        diff = (scores[m] - scores[n])**2
        diff = round(diff, 2)
        diff = fractions.Fraction(diff).limit_denominator(1000)
        terms.append(z3.If(slot, 1, 0))
        sq_terms.append(z3.If(slot, diff.numerator * 1000 / diff.denominator,
                              0))
  return z3.Sum(terms), z3.Sum(sq_terms)


try:
  file('dat')
except IOError:
  cPickle.dump(Fetch(), file('dat', 'w'))
names_and_scores, previous_pairings, requested_matches = cPickle.load(file(
    'dat'))
names_and_scores = list(reversed(names_and_scores))
requested_matches = list(reversed(requested_matches))

players = {
    name: id
    for (id, (name, score)) in zip(itertools.count(), names_and_scores)
}
scores = {
    id: score
    for (id, (name, score)) in zip(itertools.count(), names_and_scores)
}

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
  s = z3.Solver()
  s = NamedStack()
  s.push()
  slots = MakeSlots(len(players), 3)
  NoRepeatMatches(s, slots, previous_pairings)
  deadline = time.time() + seconds
  all_metrics = []
  mismatch_sum_result = [MismatchSum(slots, scores)]
  for _, squared_mismatch in mismatch_sum_result:
    all_metrics.append(squared_mismatch)
    # all_metrics.append(linear_mismatch)
  metrics = all_metrics[:]

  metric = metrics.pop(0)
  guaranteed = set()
  my_requested_matches = requested_matches[:]
  while any(my_requested_matches):
    s.push('start_round')
    for term in RequestedMatches(slots, guaranteed, my_requested_matches):
      s.add(term)

    while True:
      s.set('soft_timeout', Timeleft(deadline) * 1000)
      status = s.check()
      if status == z3.sat:
        model = s.model()
        badness = model.evaluate(metric)
        print 'Badness: {}'.format(tuple(model.evaluate(m) for m in
                                         all_metrics))
        s.push()
        if Timeleft(deadline) > 0:
          print 'Time left:', str(datetime.timedelta(seconds=Timeleft(
              deadline)))
          s.add(metric < badness)
        else:
          print 'Time limit reached.'
          s.add(metric == badness)
          break
      elif status == z3.unsat:
        try:
          model
        except NameError:
          print
          print 'You dun goofed.'
          return
        print 'OPTIMAL!'
        print 'Badness: {}'.format(tuple(model.evaluate(m) for m in
                                         all_metrics))
        s.pop()
        s.push()
        break
        ## try:
        ##   metric = metrics.pop(0)
        ##   s.push()
        ##   badness = model.evaluate(metric)
        ##   s.add(metric < badness)
        ## except IndexError:
        ##   break
      else:
        print 'Time limit reached.'
        s.pop()
        s.push()
        s.add(metric <= badness)
        break
    for n in slots:
      for m in slots[n]:
        if str(model.evaluate(slots[n][m])) == 'True' and (n,
                                                           m) not in guaranteed:
          guaranteed.add((n, m))
    s.pop('start_round')
    s.push()
    s.add([slots[n][m] for n, m in guaranteed])

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
  return z3.Or([slot != model[slot] for d in slots.values() for slot in
                d.values()])


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
        print '{:>6} {:>20} vs. {:<20} {:>6}'.format('({})'.format(scores[m]),
                                                     player, opponent,
                                                     '({})'.format(scores[n]))


def ModelPlayers(slots, model):
  for n, row in reversed(slots.items()):
    for m, playing in reversed(row.items()):
      if str(model.evaluate(playing)) == 'True':
        yield (reverse_players[m], reverse_players[n])
