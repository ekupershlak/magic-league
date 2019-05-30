# magic-league

Super-Swiss pairing algorithm for casual Magic: The Gathering leagues

At my workplace, we have a recurring Magic: the Gathering sealed league. A
problem we faced was that people needed flexibility to play varying numbers of
matches. We also wanted to maintain the benefits of Swiss pairings.

This software uses the LKH solver to solve the Traveling Salesman optimization
problem of assigning matches that respects requested numbers of matches while
minimizing pairing differences in win-percentage among the pairings. It can also
post pairings directly to a Google sheet that is specifically formatted.

## Setup

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
