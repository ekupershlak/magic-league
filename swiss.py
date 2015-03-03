from __future__ import division

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
sheets_spreadsheet = 'magic-ny FRF Sealed League'
cycle_to_pair = 3
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
      if (reverse_players[n], reverse_players[m]) in previous_pairings:
        s.add(played_0[n][m])
      else:
        s.add(z3.Not(played_0[n][m]))

  played_funcs = [played_0]
  for r, round_slots in enumerate(slots):
    if r == 0:
      continue
    played_prime = {}
    for n, row in round_slots.items():
      for m, _ in row.items():
        played_prime.setdefault(
          n, {}).setdefault(m, z3.Bool('played_{},{},{}'.format(r, n, m)))
    # Played previously means always played going forward.
    s.add([z3.Implies(played_funcs[-1][n][m], played_prime[n][m])
           for n in range(len(round_slots))
           for m in range(len(round_slots)) if n < m])
    # Add most recent round's pairings as played.
    for n, row in round_slots.items():
      for m, slot in row.items():
        if BYE in players and m == max(row) and odd(r):
          for n2 in slots[r - 1]:
            if n < max(round_slots) and n2 < max(slots[r - 1]) and n < n2:
              s.add(z3.Implies(z3.And(slots[r - 1][n2][m] and slots[r][n][m]),
                               played_prime[n][n2]))
        else:
          s.add(z3.Implies(slots[r - 1][n][m], played_prime[n][m]))
    played_funcs.append(played_prime)

  return played_funcs

def NoRepeatMatches(s, slots, played_funcs):
  for r, round_slots in enumerate(slots):
    played = played_funcs[r]
    for n, row in round_slots.items():
      for m, _ in row.items():
        s.add(z3.Implies(played[n][m], z3.Not(round_slots[n][m])))

def SignedMismatch(round_slots, scores, n, m):
  return z3.If(round_slots[n][m], scores[m] - scores[n], 0)

# Metric 1
def PerPlayerAbsoluteMismatchSumSquared(s, slots, players, scores):
  mismatches = {}
  for r, round_slots in enumerate(slots):
    for n, row in round_slots.items():
      for m, slot in row.items():
        term = z3.If(slot, scores[m] - scores[n], 0)
        mismatches.setdefault(n, []).append(term)
        mismatches.setdefault(m, []).append(term)
  def PlayersMismatchSumSquared():
    for player_id in players.values():
      term_sum = z3.Sum(mismatches[player_id])
      yield term_sum * term_sum
  return z3.Sum(list(PlayersMismatchSumSquared()))

def MismatchSum(s, slots, scores):
  for r, round_slots in enumerate(slots):
    terms = []
    sq_terms = []
    for n, row in round_slots.items():
      for m, slot in row.items():
        if BYE in players and m == max(row):
          if even(r):
            if r == len(slots) - 1:
              terms.append(z3.If(slot, scores[n], 0))
              sq_terms.append(z3.If(slot, scores[n]**2, 0))
            else:
              last_n = n
              last_slot = slot
          else:
            terms.append(z3.If(z3.And(last_slot, slot),
                               abs(scores[last_n] - scores[n]), 0))
            sq_terms.append(z3.If(z3.And(last_slot, slot),
                                  abs(scores[last_n] - scores[n]) ** 2, 0))
        else:
          terms.append(z3.If(slot, scores[m] - scores[n], 0))
          sq_terms.append(z3.If(slot, (scores[m] - scores[n]) ** 2, 0))
    yield z3.Sum(terms), z3.Sum(sq_terms)

def MaximumMismatch(s, slots, score):
  maximum = z3.Int('maximum')
  for r, round_slots in enumerate(slots):
    for n, slot in enumerate(round_slots):
      if odd(n):
        s.add(score(round_slots[n - 1]) - score(round_slots[n]) <= maximum)
  return maximum

def PerPlayerSquaredSumMismatch(s, slots, players, scores):
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


players = {name: id for (id, (score, name)) in
           zip(itertools.count(), reversed(list(itertools.chain(*groups))))}
scores = {id: score for (id, (score, name)) in
          zip(itertools.count(), reversed(list(itertools.chain(*groups))))}
if odd(len(players)):
  players[BYE] = len(players)
  scores[players[BYE]] = 0

reverse_players = {number: name for name, number in players.items()}
player_scores = {reverse_players[id]: score for (id, score) in scores.items()}

def remove_bye(l):
  return [p for p in l if p != BYE]

opponents = {}
for a, b in previous_pairings:
  if b != BYE:
    opponents.setdefault(a, []).append(b)
omw = {
  player: max(1/3., sum(player_scores[opponent] -
                        3 if BYE in opponents[player] else 0 /
                        (3 * len(remove_bye(opponents[player])))
                        for opponent in opponents[player] if opponent != BYE) /
              len(remove_bye(opponents[player])))
  for player in players if player != BYE}

def Search(seconds=180, enumeration=None):
  s = z3.Solver()
  s.push()
  slots = MakeSlots(s, len(players), 3)
  played = MakePlayedFunction(s, slots, previous_pairings, players)
  NoRepeatMatches(s, slots, played)
  all_metrics = []
  mismatch_sum_result = list(MismatchSum(s, slots, scores))
  for linear_mismatch, squared_mismatch in mismatch_sum_result:
    all_metrics.append(squared_mismatch)
    # all_metrics.append(linear_mismatch)
  #all_metrics.append(
  #  PerPlayerAbsoluteMismatchSumSquared(s, slots, players, scores))
  #all_metrics.append(PerPlayerSquaredSumMismatch(s, slots, players, scores))
  metrics = all_metrics[:]

  deadline = time.time() + seconds
  metric = metrics.pop(0)
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

  PrintModel(slots, players, scores, winning_model)
  print
  print 'Badness:', tuple(winning_model.evaluate(m) for m in all_metrics)
  return list(ModelPlayers(slots, players, scores, winning_model))

def NegateModel(slots, model):
  return z3.Or([slot != model[slot]
                for round in slots for d in round.values()
                for slot in d.values()])

def AllOptimalModels(s, slots, deadline=None):
  try:
    s.push()
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
  finally:
    s.pop()

def PrintModel(slots, players, scores, model):
  for r, round_slots in enumerate(slots):
    print 'Round', r + 1
    for n, row in reversed(round_slots.items()):
      for m, playing in reversed(row.items()):
        if str(model.evaluate(playing)) == 'True':
          player = reverse_players[m]
          opponent = reverse_players[n]
          print '{:>4} {:>20} vs. {:<20} {:>4}'.format(
            '({})'.format(scores[m]), player, opponent,
            '({})'.format(scores[n]))

def ModelPlayers(slots, players, score, model):
  for r, round_slots in enumerate(slots):
    for n, row in reversed(round_slots.items()):
      for m, playing in reversed(row.items()):
        if str(model.evaluate(playing)) == 'True':
          yield (reverse_players[m], reverse_players[n])
