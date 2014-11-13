import gspread
import itertools
import random
import sys

sheets_account = 'chrisconnett@gmail.com'
sheets_password = 'fyiyjhehvjmrongv'
sheets_spreadsheet = 'magic-ny KTK Sealed League'
n = 3

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
  for pa, pb, winner in zip(a, b, winners):
    total_mismatch[pa] += partial_scores[pb] - partial_scores[pa]
    total_mismatch[pb] += partial_scores[pa] - partial_scores[pb]
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

#assert partial_scores == dict(zip(names, scores))

groups = [list(group) for score, group in
          itertools.groupby(sorted(zip(scores, names), reverse=True))]

pairings = []

## while groups:
##   current = groups.pop()
##   random.shuffle(current)
##   while len(current) > 1:
##     pairing = (current[0], current[1])
##     if pairing has already played:
##       search group for someone [0] has not played then someone [1] has not played
##       if none go back to shuffling. in the limit, they will be in the first pairing
##       if this is the first pairing, pair down at random
## pairings.append()


##   if odd group:
##     if bottom player is ineligible for pair-down:
##       reshuffle
##     grab a random player eligible for pair-up
