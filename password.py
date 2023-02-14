"""OAuth2 authentication for magic-ny league parings."""

import gspread
from google.auth import default

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]


def GetGc():
  credentials, _ = default(scopes=scope)
  return gspread.authorize(credentials)
