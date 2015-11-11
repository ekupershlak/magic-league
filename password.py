"""OAuth2 authentication for magic-ny league parings."""

import json

import gspread
from oauth2client.client import SignedJwtAssertionCredentials

json_key = json.load(open('magic-ny-pairings.json'))
scope = ['https://spreadsheets.google.com/feeds']

credentials = SignedJwtAssertionCredentials(json_key['client_email'],
                                            json_key['private_key'], scope)
gc = gspread.authorize(credentials)
