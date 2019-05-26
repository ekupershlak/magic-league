# python3
"""Player datatype."""
import fractions
from typing import Collection, NamedTuple, Text

Name = Text
Username = Text


class Player(NamedTuple):
  id: Username
  name: Name
  score: fractions.Fraction
  requested_matches: int
  opponents: Collection[Username]
