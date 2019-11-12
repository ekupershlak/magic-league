# magic-league

A Super-Swiss pairing algorithm for casual Magic: The Gathering leagues.

At my workplace, we have a recurring Magic: the Gathering sealed league. A
problem we faced was that people needed flexibility to play varying numbers of
matches. We also wanted to maintain the benefits of Swiss pairings.

This software uses the LKH traveling salesman solver to assign matches in a way
that respects requested numbers of matches while minimizing differences in
win-percentage among the pairings. It can post those pairings directly to a
Google sheet that is specifically formatted.

## Setup

**Python 3 is required.**

### Install prerequisites

```
pip install -r requirements.txt
```

### Create `credentials.json`

To connect to a spreadsheet for automatic import and export, you'll need to
create a Google Cloud service account and populate `credentials.json`.

1.  Create a service account.
    https://developers.google.com/android/management/service-account
1.  Select `Create credentials > Server account key`.
1.  Select your service account and download a credentials file in JSON format.
1.  Save that file in the project root directory as `credentials.json`.
1.  Give the account write access to the league Sheet through Google Sheets's
    sharing interface. Share with the email address found in `credentials.json`
    under `client_email`.

## Run

The most common invocation:

```
python3 swiss.py <set code> <cycle number> -w
```

or `--help` for all the options.

## Template League Spreadsheet

The cells from which to read past pairings and to which to post new pairings are
hard-coded in `sheet_manager.py`. You can use [this template sheet][1] to track
your own league, or adapt `sheet_manager` to use ranges appropriate to an
existing sheet.

[1]: https://docs.google.com/spreadsheets/d/1wDgi1rTJ3bq7-i91jEPzho4gVGx2SAaKOSALNtz41CA/edit?usp=sharing
