# python3
"""Liaison to Google Sheets."""

from abc import ABC, abstractmethod
from collections import Counter
from typing import List
import datetime
import fractions
import hashlib
import itertools
import os
import pickle
import random

import gspread
import magic_sets
import password
import player as player_lib

DEADLINE_WEEKDAY = 4  # Friday


def CycleDeadline():
    today = datetime.date.today()
    weekday = today.weekday()
    cycle_length = (DEADLINE_WEEKDAY - weekday) % 7 + 7
    if cycle_length < 10:
        cycle_length += 7
    deadline = today + datetime.timedelta(days=cycle_length)
    return deadline


class SheetManager(ABC):
    """Marshals data to and from the Google Sheet."""

    def __init__(self, cycle, fetch: bool):
        self.cycle = cycle
        self.sheet: gspread.Spreadsheet = None
        self.fetch = fetch

    @abstractmethod
    def _ConnectToSheet(self):
        pass

    @property
    @abstractmethod
    def _filename(self) -> str:
        pass

    def GetPlayers(self):
        player_list = self._FetchFromCache()
        random.shuffle(player_list)
        return player_list

    def Writeback(self, pairings):
        """Write the pairings to the sheet."""
        # Some aspect of the connection to the spreadsheet can go stale. Reload it
        # just before writing to make sure it's fresh.
        self._ConnectToSheet()

        last_pairing_row = len(pairings) + 1

        ws_name = "Cycle " + str(self.cycle)
        output = self.sheet.worksheet(ws_name)

        # Delete extra rows.
        if last_pairing_row < output.row_count:
            output.delete_rows(last_pairing_row + 1, output.row_count)
        # Copy formulas from the first row to all subsequent rows.
        output.copy_range("2:2", f"3:{last_pairing_row}")
        # Add pairings.
        pairings_range = output.range(f"B2:C{last_pairing_row}")

        flattened_pairings = itertools.chain.from_iterable(pairings)
        for cell, player in zip(pairings_range, flattened_pairings):
            cell.value = player.name
        print("Writing to", ws_name)
        output.update_acell("I1", CycleDeadline().strftime("%B %d"))
        output.update_cells(pairings_range)

    def _FetchFromCache(self):
        """Fetches data from local file, falling back to the spreadsheet."""

        try:
            mtime = datetime.datetime.fromtimestamp(
                os.stat(self._filename).st_mtime)
            age = datetime.datetime.now() - mtime
            if age < datetime.timedelta(minutes=20) and not self.fetch:
                player_list = pickle.load(open(self._filename, "rb"))
                print("Loaded previous results from cache")
                return player_list
        except (IOError, EOFError, FileNotFoundError):
            pass
        player_list = self._FetchFromSheet()
        pickle.dump(player_list, open(self._filename, "wb"))
        return player_list

    def _FetchFromSheet(self):
        """Fetches data from the spreadsheet."""
        print("Fetching from spreadsheet…")
        self._ConnectToSheet()
        standings = self.sheet.worksheet("Standings")
        names = list(standings.col_values(1)[1:])
        ids = list(standings.col_values(2)[1:])
        wins = [int(n) for n in standings.col_values(4)[1:]]
        losses = [int(n) for n in standings.col_values(5)[1:]]
        requested_matches = [
            int(s) for s in standings.col_values(10 + self.cycle - 1)[1:]
        ]
        scores = [
            fractions.Fraction(n_win + 1, n_win + n_loss + 2)
            for n_win, n_loss in zip(wins, losses)
        ]

        previous_pairings = set()
        for i in range(1, self.cycle):
            cycle_sheet = self.sheet.worksheet(f"Cycle {i}")
            a = cycle_sheet.col_values(1)[1:]  # type: List[Username]
            b = cycle_sheet.col_values(4)[1:]  # type: List[Username]
            previous_pairings |= set(zip(a, b))
            previous_pairings |= set(zip(b, a))

        player_list = []
        for vitals in zip(ids, names, scores, requested_matches):
            id_, name, score, rm = vitals
            opponent_ids = tuple(b for (a, b) in previous_pairings if id_ == a)
            player_list.append(player_lib.Player(
                id_, name, score, rm, opponent_ids))
        print("Fetched previous results from sheet")
        _validate_players(player_list)
        return player_list


def _validate_players(players: List[player_lib.Player]):
    player_ids = Counter([player.id for player in players])
    duplicates = [k for (k, v) in player_ids.items() if v > 1]
    if len(duplicates) > 0:
        raise DuplicatePlayerError(f'Duplicate player IDs: {player_ids}')


class DuplicatePlayerError(Exception):
    """Error due to duplicate players."""


class SetSheetManager(SheetManager):
    """Marshals data to and from the Google Sheet."""

    def __init__(self, set_code, cycle, fetch: bool):
        super().__init__(cycle, fetch)
        self.set_code = set_code

    def _ConnectToSheet(self):
        self.sheet = password.GetGc().open(
            f"magic-ny {magic_sets.names[self.set_code]} ({self.set_code}) Sealed League"
        )
        print(f"Using sheet {self.sheet.id} for set {self.set_code}")

    @property
    def _filename(self) -> str:
        return f"{self.set_code}-{self.cycle}"


class UrlSheetManager(SheetManager):
    """Marshals data to and from the Google Sheet."""

    def __init__(self, url, cycle, fetch: bool):
        super().__init__(cycle, fetch)
        self.url = url

    def _ConnectToSheet(self):
        self.sheet = password.GetGc().open_by_url(self.url)
        print(f"Using sheet {self.sheet.id} by URL")

    @property
    def _filename(self) -> str:
        h = hashlib.sha256()
        h.update(self.url.encode('utf8'))
        return f"{h.hexdigest()}-{self.cycle}"
