"""Randomly pairs players for initial rounds of Magic league."""

import random


def PreviousOpponents(player, rounds):
  for r in rounds:
    for pairing in r:
      if player in pairing:
        yield list(set(pairing) - set([player]))[0]


def MakeRound(players, rounds):
  players = players[:]
  while len(players) >= 2:
    player = random.choice(players)
    players.remove(player)
    opp = random.choice(list(set(players) - set(PreviousOpponents(player,
                                                                  rounds))))
    players.remove(opp)
    yield (player, opp)
  if players:
    yield (list(players)[0], 'BYE')


players = [
('Alan Strohm', 0),
('Jeremy Espenshade', 3),
('Joseph Wen', 2),
('Son Nguyen', 3),
('Alex Ogier', 2),
('Edward Kupershlak', 2),
('Ryan Murphy', 3),
('Alek Dembowski', 1),
('Brooklyn Schlamp', 2),
('Michael Rubin', 3),
('RJ Sumi', 0),
('Shifan Mahroof', 3),
('Charles Macanka', 0),
('David Goldberg', 3),
('Jen Bonczar', 2),
('Joseph Prete', 2),
('Phil Koonce', 2),
('Ram Dobson', 3),
('Timothy Yuan', 2),
('Dalton Petursson', 3),
('Daniel Schneider', 3),
('Dylan Shinzaki', 3),
('Fabio Drucker', 3),
('Hesky Fisher', 3),
('Nunzio Thron', 3),
('Rob Longstaff', 3),
('Todd Layton', 2),
('Chris Connett', 0),
('David Katz', 1),
('George Reis', 0),
('Herve Bronnimann', 3),
('Jonathan Jarvis', 1),
('Michael Sobin', 3),
('Noah Broestl', 2),
('Scott Zibble', 3),
('Zhuoran Yu', 2),
('Zlata Barshteyn', 1),
]

rounds = [
[('Dalton Petursson', 'Joseph Prete'),
('David Katz', 'Dylan Shinzaki'),
('Edward Kupershlak', 'George Reis'),
('Fabio Drucker', 'Ryan Murphy'),
('Joseph Wen', 'Phil Koonce'),
('Noah Broestl', 'Son Nguyen'),
('Brooklyn Schlamp', 'David Goldberg'),
('Scott Zibble', 'Todd Layton'),
('Daniel Schneider', 'Ram Dobson'),
('Jeremy Espenshade', 'Michael Rubin'),
('RJ Sumi', 'Shifan Mahroof'),
('Jen Bonczar', 'Jonathan Jarvis'),
('Alan Strohm', 'Timothy Yuan'),
('Rob Longstaff', 'Zhuoran Yu'),
('Hesky Fisher', 'Nunzio Thron'),
('Alex Ogier', 'Chris Connett'),
('Alek Dembowski', 'Zlata Barshteyn'),
('Jeremy Espenshade', 'Charles Macanka'),
('Alan Strohm', 'David Katz'),
('Daniel Schneider', 'Jeremy Espenshade'),
('Alan Strohm', 'Todd Layton'),
('Joseph Wen', 'Zhuoran Yu'),
('Noah Broestl', 'RJ Sumi'),
('George Reis', 'Timothy Yuan'),
('Phil Koonce', 'Scott Zibble'),
('Edward Kupershlak', 'Fabio Drucker'),
('Charles Macanka', 'Rob Longstaff'),
('Alex Ogier', 'Shifan Mahroof'),
('Ram Dobson', 'Ryan Murphy'),
('Dylan Shinzaki', 'Michael Rubin'),
('Dalton Petursson', 'Jen Bonczar'),
('Brooklyn Schlamp', 'Chris Connett'),
('David Katz', 'Hesky Fisher'),
('Joseph Prete', 'Son Nguyen'),
('David Goldberg', 'Nunzio Thron'),
('Son Nguyen', 'Zhuoran Yu'),
('Ram Dobson', 'Scott Zibble'),
('Dalton Petursson', 'Nunzio Thron'),
('Dylan Shinzaki', 'Fabio Drucker'),
('Hesky Fisher', 'Michael Rubin'),
('Daniel Schneider', 'Shifan Mahroof'),
('RJ Sumi', 'Rob Longstaff'),
('Joseph Wen', 'Todd Layton'),
('David Goldberg', 'Ryan Murphy'),
 ] ]
print list(MakeRound(players, []))

for i in range(3):
  rounds.append(list(MakeRound(
      [player for (player, n) in players if i < n], rounds)))

for i, round in enumerate(rounds[1:]):
  print
  print 'Round', i
  for a, b in round:
    if a > b:
      a, b = b, a
    print '{}\t{}'.format(a, b)
