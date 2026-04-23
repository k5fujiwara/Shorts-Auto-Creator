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

The workflow runs three times per day using UTC:

- `14:00 UTC` for a US morning window
- `18:00 UTC` for a US midday window
- `02:00 UTC` for a US evening window

It also supports manual runs through `workflow_dispatch`.

## Notes

- `posted_articles.json` is committed back to the repository after successful runs.
- BGM files must exist in the repository under `bgm/`.
- If you refresh YouTube auth locally and get a new `token.pickle`, update the matching GitHub secret.
