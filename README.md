# magic-league

A Super-Swiss pairing algorithm for casual Magic: The Gathering leagues.

At my workplace, we have a recurring Magic: the Gathering sealed league. A
problem we faced was that people needed flexibility to play varying numbers of
matches. We also wanted to maintain the benefits of Swiss pairings.

This software uses the min-weight graph matching to assign matches in a way that
respects requested numbers of matches while minimizing differences in
win-percentage among the pairings. It can post those pairings directly to a
Google sheet that is specifically formatted.

## Setup

### Install prerequisites

```
pip install -r requirements.txt
```

### Create a service account and `credentials.json`

To connect to a spreadsheet for automatic import and export, you'll need a
Google Cloud project with a service account. Then we'll download its
credentials.

1.  Select or create a Cloud project. https://console.cloud.google.com/project

1.  Enable Sheets API and Drive API at
    https://console.cloud.google.com/apis/library . Search for "Sheets", then
    separately "Drive", and enable each.

1.  Now go to https://console.cloud.google.com/apis/credentials . If you don't
    have a dedicated service account for this exact purpose yet (posting
    pairings), create one through `Create credentials > Server account`.

1.  Select your service account from the `Service accounts` section.

    1.  Navigate to the `KEYS` tab.
    1.  Select `ADD KEY` and `Create new key` to download a credentials file in
        JSON format.

1.  Save that file in the project root directory as `credentials.json`.

1.  Give the account write access to the league Sheet through Google Sheets's
    sharing interface by sharing it with the email address found in
    `credentials.json` under `client_email`.


## Local authentication

Use `gcloud` to authenticate as yourself:

```sh
gcloud auth application-default login \
    --scopes=https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform \
    --project <quota project>
```

Then configure the credential file as the default:

```sh
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json
```

## Run

The most common invocation:

```
python swiss.py <set code> <cycle number> -w
```

or `--helpshort` for all the options.

## Template League Spreadsheet

The cells from which to read past pairings and to which to post new pairings are
hard-coded in `sheet_manager.py`. You can use [this template sheet][1] to track
your own league, or adapt `sheet_manager` to use ranges appropriate to an
existing sheet.

## (Cloud) Run

As an alternative, this script can be hosted using Google Cloud Run. 

### Develop

To develop locally, set an application-default credential with the requisite permissions:

```
gcloud auth application-default login \
  --scopes https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/spreadsheets
```

Then run the function:

```
functions-framework --target=generate_pairing
```

### Deploy

Deploy it using the following command:

```
gcloud functions deploy <CloudRun-func-name> --gen2 \
    --runtime=python311 --source=. --entry-point=generate_pairings \
    --trigger-http \
    --region=<region> --project=<project>
```


[1]: https://docs.google.com/spreadsheets/d/1wDgi1rTJ3bq7-i91jEPzho4gVGx2SAaKOSALNtz41CA/edit?usp=sharing
