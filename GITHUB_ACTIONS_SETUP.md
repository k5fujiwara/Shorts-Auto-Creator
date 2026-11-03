# GitHub Actions Setup

GitHub Actions uses repository secrets to restore the files needed at runtime.

## Required repository secrets

- `APP_ENV_BASE64`
- `CLIENT_SECRETS_JSON_BASE64`
- `YOUTUBE_TOKEN_PICKLE_BASE64`

## PowerShell commands to create secret values

Run these in the project root on your Windows machine.

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes(".env"))
```

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("client_secrets.json"))
```

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("token.pickle"))
```

Copy each output and paste it into the matching GitHub repository secret.

## Workflow schedule

The workflow runs every 4 hours using UTC, for up to 6 runs per day:

- `0 */4 * * *`

It also supports manual runs through `workflow_dispatch`.

## Notes

- `posted_articles.json` is committed back to the repository after successful runs.
- BGM files must exist in the repository under `bgm/`.
- If you refresh YouTube auth locally and get a new `token.pickle`, update the matching GitHub secret.
- YouTube video uploads consume a large amount of API quota. Six uploads per day is close to the default daily quota, so monitor failures and request a higher YouTube Data API quota if needed.

## Recovering from YouTube token expiration

If GitHub Actions fails with the following error, the YouTube OAuth token has expired or was revoked:

```text
google.auth.exceptions.RefreshError: ('invalid_grant: Token has been expired or revoked.'
```

Create a new `token.pickle` locally and update the GitHub secret.

Run these commands in PowerShell from the project root:

```powershell
cd "C:\Users\keigo\OneDrive\デスクトップ\自動化ツール\Shorts-Auto-Creator"
Remove-Item token.pickle
python -c "from youtube_uploader import get_authenticated_youtube_service; get_authenticated_youtube_service()"
Test-Path token.pickle
```

When the browser opens, sign in with the YouTube account used for uploads and approve access.
`Test-Path token.pickle` should return `True`.

Convert the new token to Base64:

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\keigo\OneDrive\デスクトップ\自動化ツール\Shorts-Auto-Creator\token.pickle"))
```

Copy the full output and update the GitHub repository secret:

`Settings` -> `Secrets and variables` -> `Actions` -> `YOUTUBE_TOKEN_PICKLE_BASE64` -> `Update`

After updating the secret, run the `Shorts Automation` workflow manually from the `Actions` tab and confirm it succeeds.
