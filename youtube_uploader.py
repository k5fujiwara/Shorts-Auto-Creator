#!/usr/bin/env python
"""Upload a generated Shorts video to YouTube."""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from video_generator import (
    Article,
    cleanup_model_text,
    fetch_latest_article,
    generate_with_retry,
    get_client,
    resolve_source_url,
)


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


@dataclass
class UploadMetadata:
    title: str
    description: str


@dataclass
class UploadResult:
    video_id: str
    title: str
    description: str
    response: dict[str, Any]


def normalize_title(raw_title: str) -> str:
    cleaned = cleanup_model_text(raw_title)
    cleaned = " ".join(cleaned.replace("\n", " ").split())
    cleaned = cleaned.strip(" \"'`#")
    cleaned = cleaned.replace("#Shorts", "").strip()
    if not cleaned:
        cleaned = "Quick Learning Boost"

    if len(cleaned) > 60:
        cleaned = cleaned[:57].rstrip(" ,.!?-:;") + "..."

    final_title = f"{cleaned} #Shorts"
    return final_title[:100].rstrip()


def generate_upload_metadata(
    article: Article,
    narration: str,
    client=None,
    title_model: str | None = None,
) -> UploadMetadata:
    client = client or get_client()
    model_name = title_model or os.environ.get(
        "GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"
    )

    title_prompt = f"""
Write one compelling YouTube Shorts title in English.

Requirements:
- Around 30 characters if possible
- Hook curiosity fast
- No quotes
- No emojis
- Do not include #Shorts
- Output only the title

Article title:
{article.title}

Narration:
{narration[:1200]}
""".strip()

    summary_prompt = f"""
Write a concise English YouTube description summary for this short video.

Requirements:
- 2 short sentences
- Clear and practical
- No bullet points
- No hashtags
- Do not repeat the URL

Article title:
{article.title}

Article body:
{article.body[:3000]}

Narration:
{narration[:1200]}
""".strip()

    title_response = generate_with_retry(
        client=client,
        model_name=model_name,
        prompt=title_prompt,
    )
    summary_response = generate_with_retry(
        client=client,
        model_name=model_name,
        prompt=summary_prompt,
    )

    title = normalize_title(title_response.text or "")
    summary = cleanup_model_text(summary_response.text or "")
    summary = " ".join(summary.replace("\n", " ").split()).strip()
    if not summary:
        summary = narration

    description = f"{article.link}\n\n{summary}"
    return UploadMetadata(title=title, description=description)


def get_authenticated_youtube_service(
    client_secrets_path: str | Path = "client_secrets.json",
    token_path: str | Path = "token.pickle",
):
    secrets_path = Path(client_secrets_path)
    token_file = Path(token_path)

    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Client secrets file not found: {secrets_path}"
        )

    credentials = None
    if token_file.exists():
        with token_file.open("rb") as handle:
            credentials = pickle.load(handle)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
        credentials = flow.run_local_server(port=0)
        with token_file.open("wb") as handle:
            pickle.dump(credentials, handle)

    return build("youtube", "v3", credentials=credentials)


def upload_short(
    video_path: str | Path,
    article: Article,
    narration: str,
    *,
    client=None,
    client_secrets_path: str | Path = "client_secrets.json",
    token_path: str | Path = "token.pickle",
    title_model: str | None = None,
    privacy_status: str = "private",
) -> UploadResult:
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    metadata = generate_upload_metadata(
        article=article,
        narration=narration,
        client=client,
        title_model=title_model,
    )
    youtube = get_authenticated_youtube_service(
        client_secrets_path=client_secrets_path,
        token_path=token_path,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "categoryId": "27",
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(str(video_file), chunksize=-1, resumable=True),
    )

    response = None
    while response is None:
        _, response = request.next_chunk()

    return UploadResult(
        video_id=response["id"],
        title=metadata.title,
        description=metadata.description,
        response=response,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a generated Shorts video.")
    parser.add_argument("--video-path", default="output.mp4")
    parser.add_argument(
        "url",
        nargs="?",
        help="RSS URL or site URL. If omitted, NOTE_URL in .env is used.",
    )
    parser.add_argument("--narration-file", default="narration.txt")
    parser.add_argument(
        "--client-secrets",
        default=os.environ.get("YOUTUBE_CLIENT_SECRETS", "client_secrets.json"),
    )
    parser.add_argument(
        "--token-file",
        default=os.environ.get("YOUTUBE_TOKEN_FILE", "token.pickle"),
    )
    parser.add_argument(
        "--privacy-status",
        choices=("private", "public", "unlisted"),
        default=os.environ.get("YOUTUBE_PRIVACY_STATUS", "private"),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_url = resolve_source_url(args.url)
    article = fetch_latest_article(source_url)
    narration = Path(args.narration_file).read_text(encoding="utf-8").strip()
    result = upload_short(
        video_path=args.video_path,
        article=article,
        narration=narration,
        client=get_client(),
        client_secrets_path=args.client_secrets,
        token_path=args.token_file,
        privacy_status=args.privacy_status,
    )
    print("Upload completed:")
    print(f"- video_id: {result.video_id}")
    print(f"- title: {result.title}")


if __name__ == "__main__":
    main()
