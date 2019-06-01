"""OAuth2 authentication for magic-ny league parings."""

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]


def GetGc():
  credentials = ServiceAccountCredentials.from_json_keyfile_name(
      'credentials.json', scope)
  return gspread.authorize(credentials)
