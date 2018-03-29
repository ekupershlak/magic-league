# -*- encoding: utf-8 -*-
"""Generate cube packs according to a Wizards-like collation."""

from __future__ import print_function

import collections
import random

N = 24
PACK_SIZE = 15

dist = collections.defaultdict(int)

strains = '12345ABC'
primaries = strains[:5]


def GenPack():
  """Generate a pack as a string representing what piles to pull from."""
  pack = []
  while len(pack) < PACK_SIZE:
    # Add cards to a pack based on print runs of length k.
    k = random.randint(4, 8)

    # An association of strains and their desired relative as-fan.
    cube = dict(zip(strains, [75, 75, 75, 75, 75, 75, 75, 30]))

    while k and len(pack) < PACK_SIZE:
      k -= 1
      # Pick a strain based on the desired as-fan.
      x = random.choices(list(cube.keys()), list(cube.values()))[0]
      # If it's the same strain as the last run ended on, start over.
      if pack and pack[-1] == x:
        break
      # Disallow this strain from being used again in this run.
      del cube[x]
      # Don't allow the pack to end on the same strain it started on.  We're
      # about to rotate the pack to put a monocolor card in front, and we don't
      # want a repeated strain added by that process.
      if len(pack) == PACK_SIZE - 1 and pack[0] == x:
        break
      pack.append(x)

  # Count the pack for statistics.
  for card in pack:
    dist[card] += 1

  # Rotate the pack until a monocolor is in the first position. This guarantees
  # each pack opens with a monocolor card.
  while pack[0] not in primaries:
    # Thm: This loop terminates.
    #
    # Pf: The minimum run length is 4 and there are only 3 non-primary
    # strains. Every run is guaranteed to have a primary entry. Because every
    # pack has at least one run (actually at least two), every pack has a
    # primary entry. Eventually that entry will be at the front of the list and
    # this loop will terminate. ∎
    shift = pack.pop()
    pack[0:0] = shift

  return ' '.join(''.join(pack[i:i + 3]) for i in range(0, PACK_SIZE, 3))


for _ in range(N // 3):
  print(GenPack())
  print(GenPack())
  print(GenPack())
  print()

print('As-fan')
print('-----')
for strain, count in sorted(dist.items(), key=lambda kv: strains.find(kv[0])):
  print('{}: {:.2f}'.format(strain, count / N))
