"""Solver for swiss pairings."""

from __future__ import print_function

import collections
import datetime
import fractions
import itertools
import pickle
import random
import time

import parallelize
import z3

import password

BYE = 'BYE'


class NamedStack(z3.Solver):
  """A z3.Solver that supports pushing and popping to named frames."""

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


def MakeSlots(n_players):
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
  if n in (1, 2, 3):
    return EnumeratedPopCount(vs, n)
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


def RequestedMatches(slots, requested_matches, reverse_players):
  """Guarantees players get their requested number of matches.

  Args:
    slots: slot variables
    requested_matches: the number of matches each player has requested
    reverse_players: the reverse_players dict
  Yields:
    Terms over slots (to be added to a Solver) that guarantees players have
    their requested number of matches.
  """

  n_players = len(slots) + 1

  order = sorted(
      range(n_players), key=lambda n: requested_matches[n], reverse=True)
  for n in parallelize.parallelize(order):
    print(reverse_players[n], 'requests', requested_matches[n], 'matches')
    n_adjacency = []
    for m in range(n_players):
      if n < m:
        n_adjacency.append(slots[n][m])
      elif n > m:
        n_adjacency.append(slots[m][n])
    yield PopCount(n_adjacency, requested_matches[n])


def NoRepeatMatches(s, slots, previous_pairings, reverse_players):
  for n, row in list(slots.items()):
    for m, _ in list(row.items()):
      if (reverse_players[n], reverse_players[m]) in previous_pairings:
        s.add(z3.Not(slots[n][m]))


def MismatchSum(slots, scores):
  """Terms for sum of mismatch and squared mismatch."""
  terms = []
  sq_terms = []
  for n, row in list(slots.items()):
    for m, slot in list(row.items()):
      if n < m:
        diff = (scores[m] - scores[n])**2
        diff = round(diff, 2)
        diff = fractions.Fraction(diff).limit_denominator(1000)
        terms.append(z3.If(slot, 1, 0))
        sq_terms.append(z3.If(slot, diff.numerator * 1000 / diff.denominator,
                              0))
  return z3.Sum(terms), z3.Sum(sq_terms)


class Pairer(object):
  """Manages pairing a cycle of a league."""

  def __init__(self, set_code, cycle):
    self.set_code = set_code
    self.cycle = cycle

    names_scores_matches, self.previous_pairings = self._Fetch()
    self.players = {
        name: id
        for (id, (name, _, _)) in zip(itertools.count(), names_scores_matches)
    }

    self.scores = {
        id: score
        for (id, (_, score, _)) in zip(itertools.count(), names_scores_matches)
    }
    self.requested_matches = {
        id: m
        for (id, (_, _, m)) in zip(itertools.count(), names_scores_matches)
    }

    self.reverse_players = {number: name
                            for name, number in list(self.players.items())}
    self.player_scores = {self.reverse_players[id]: score
                          for (id, score) in list(self.scores.items())}

  def Search(self, seconds=3600):
    """Constructs an SMT problem for pairings and solves it."""
    s = z3.Solver()
    s = NamedStack()
    s.push()
    slots = MakeSlots(len(self.players))
    NoRepeatMatches(s, slots, self.previous_pairings, self.reverse_players)
    deadline = time.time() + seconds
    _, metric = MismatchSum(slots, self.scores)

    for term in RequestedMatches(slots, self.requested_matches,
                                 self.reverse_players):
      s.add(term)

    while True:
      s.set('soft_timeout', Timeleft(deadline) * 1000)
      status = s.check()
      if status == z3.sat:
        model = s.model()
        badness = model.evaluate(metric)
        print('Badness: {}'.format(tuple(model.evaluate(metric))))
        s.push()
        if Timeleft(deadline) > 0:
          print('Time left:',
                str(datetime.timedelta(seconds=Timeleft(deadline))))
          s.add(metric < badness)
        else:
          print('Time limit reached.')
          s.add(metric == badness)
          break
      elif status == z3.unsat:
        try:
          model
        except NameError:
          print()
          print('You dun goofed.')
          return
        print('OPTIMAL!')
        print('Badness: {}'.format(tuple(model.evaluate(metric))))
        s.pop()
        s.push()
        break
      else:
        print('Time limit reached.')
        s.pop()
        s.push()
        s.add(metric <= badness)
        break

    self._PrintModel(slots, self.scores, model)
    print()
    print('Badness:', model.evaluate(metric))
    return list(self.ModelPlayers(slots, model))

  def Writeback(self, pairings):
    spreadsheet = self.GetSpreadsheet()
    ws_name = 'Cycle ' + str(self.cycle)
    output = spreadsheet.worksheet(ws_name)
    pairings_range = output.range('B2:C' + str(len(pairings) + 1))
    for cell, player in zip(pairings_range,
                            (player for row in pairings for player in row)):
      cell.value = player
    print('Writing to', ws_name)
    output.update_cells(pairings_range)

  def _Fetch(self, from_cache=True):
    """Fetches data from local file, falling back to the spreadsheet."""

    if from_cache:
      try:
        return pickle.load(open(self.set_code))
      except IOError:
        pass
    names_scores_matches, previous_pairings = self._FetchFromSheet()
    pickle.dump(
        (names_scores_matches, previous_pairings), open(self.set_code, 'w'))

    return names_scores_matches, previous_pairings

  def _FetchFromSheet(self):
    """Fetches data from the spreadsheet."""

    spreadsheet = self.GetSpreadsheet()
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
    print('lcm is', lcm)

    requested_matches = [
        int(s) for s in standings.col_values(9 + self.cycle - 1)[1:]
    ]
    names, requested_matches = list(zip(*[(
        n, rm) for (n, rm) in zip(names, requested_matches) if 0 < rm <= 3]))
    names = list(names)
    requested_matches = list(requested_matches)
    print(list(zip(names, requested_matches)))

    previous_pairings = set()

    for i in range(1, self.cycle):
      cycle = spreadsheet.worksheet('Cycle {}'.format(i))
      a = cycle.col_values(2)[1:]
      b = cycle.col_values(3)[1:]

      previous_pairings |= set(zip(a, b))
      previous_pairings |= set(zip(b, a))

    if Odd(sum(requested_matches)):
      targetted_for_bye = 3
      candidates = [
          (i, name)
          for i, (name, n_requested) in enumerate(zip(names, requested_matches))
          if n_requested == targetted_for_bye and (name, BYE
                                                  ) not in previous_pairings
      ]
      byed_i, byed_name = random.choice(candidates)
      requested_matches[byed_i] -= 1
      print(byed_name, 'receives a bye.')

    names_scores_matches = list(zip(names, scores, requested_matches))
    random.shuffle(names_scores_matches)
    return names_scores_matches, previous_pairings

  def GetSpreadsheet(self):
    return password.gc.open('magic-ny {} Sealed League'.format(self.set_code))

  def PrintModel(self, slots, model):
    for n, row in reversed(list(slots.items())):
      for m, playing in reversed(list(row.items())):
        if str(model.evaluate(playing)) == 'True':
          player = self.reverse_players[m]
          opponent = self.reverse_players[n]
          print('{:>6} {:>20} vs. {:<20} {:>6}'.format(
              '({})'.format(self.scores[m]), player, opponent,
              '({})'.format(self.scores[n])))

  def ModelPlayers(self, slots, model):
    for n, row in reversed(list(slots.items())):
      for m, playing in reversed(list(row.items())):
        if str(model.evaluate(playing)) == 'True':
          yield (self.reverse_players[m], self.reverse_players[n])
