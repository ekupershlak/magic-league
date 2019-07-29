# python3
"""Implementation of the Blitzstein-Diaconis Graph Algorithm.

http://www.people.fas.harvard.edu/~blitz/BlitzsteinDiaconisGraphAlgorithm.pdf
"""

import fractions
import random
from typing import Collection, Tuple

Graph = Collection[Tuple[int, int]]


def Graphical(d):
  """Whether the given degree sequence is graphical."""
  n = len(d)
  if any(di < 0 for di in d):
    return False
  d = [di for di in d if di > 0]
  d.sort(reverse=True)
  for k in range(1, n + 1):
    if sum(d[:k]) > k * (k - 1) + sum(min(k, di) for di in d[k:]):
      return False
  return True


def ArrayDecrement(indices, array):
  """Returns *copy of* `array`, having decremented each index in `indices`."""
  a = array[:]
  for i in indices:
    a[i] -= 1
  return a


def ImportanceSampledBlitzsteinDiaconis(d, n=100) -> Graph:
  """Sample from `n` graphs according to their probability of generation.

  See §8 of
  http://www.people.fas.harvard.edu/~blitz/BlitzsteinDiaconisGraphAlgorithm.pdf

  Args:
    d: a graphical degree sequence
    n: size of pool from which the final graph will be sampled by importance

  Returns:
    A graph with degree sequence `d`.
  Raises:
    ValueError: if d is not graphical
  """
  population = []
  weights = []
  for _ in range(n):
    e, c_sigma = BlitzsteinDiaconis(d)
    population.append(e)
    weights.append(1 / c_sigma)
  return random.choices(population, weights, k=1)[0]


def BlitzsteinDiaconis(d) -> Tuple[fractions.Fraction, Graph]:
  """Generates a random graph with degree sequence `d`.

  See §4 of
  http://www.people.fas.harvard.edu/~blitz/BlitzsteinDiaconisGraphAlgorithm.pdf

  Args:
    d: a graphical degree sequence

  Returns:
    Pair of (edge-set of a graph with degree sequence `d`,
             relative liklihood of selecting this graph uniformly).
  Raises:
    ValueError: if d is not graphical
  """
  d = d[:]
  equivalence_class_size = 1
  likelihood = fractions.Fraction(1)
  # 1. Let E be an empty list of edges.
  e = set()
  if not Graphical(d):
    raise ValueError(f'{d} is not graphical.')
  # 2. If d = 0, terminate with output E.
  while any(di > 0 for di in d):
    # 3. Choose the least i with di a minimal positive entry.
    minimum = min(di for di in d if di > 0)
    i = d.index(minimum)
    equivalence_class_size *= minimum
    # 7. Repeat steps 4-6 until the degree of i is 0.
    while d[i] > 0:
      # 4. Compute candidate list J = {j ≠ i : {i, j} ∉ E and ⊖ᵢ,ⱼ d is
      # graphical}
      candidates = {
          j for j in range(len(d)) if j != i and tuple(sorted((
              i, j))) not in e and Graphical(ArrayDecrement((i, j), d))
      }
      # 5. Pick j ∈ J with probability proportional to its degree in d.
      selection = random.choices(
          list(candidates), [d[j] for j in candidates], k=1)[0]
      likelihood *= d[selection]
      likelihood /= sum(d[j] for j in candidates)
      # 6. Add the edge {i, j} to E and update d to ⊖ᵢ,ⱼ d.
      e.add(tuple(sorted((i, selection))))
      d = ArrayDecrement((i, selection), d)
      assert Graphical(d)
    # 8. Return to step 2.
  # print(f'c(Y) = {equivalence_class_size}')
  # print(f'σ(Y) = {likelihood}')
  return e, likelihood * equivalence_class_size
