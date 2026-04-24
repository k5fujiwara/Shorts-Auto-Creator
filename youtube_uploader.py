#!/usr/bin/env python
"""Upload a generated Shorts video to YouTube."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
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


def normalize_summary(raw_summary: str) -> str:
    summary = cleanup_model_text(raw_summary)
    summary = " ".join(summary.replace("\n", " ").split()).strip()
    return summary


def extract_first_sentences(text: str, limit: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    picked = [sentence.strip() for sentence in sentences if sentence.strip()][:limit]
    return " ".join(picked).strip()


def build_fallback_title(narration: str) -> str:
    sentence = extract_first_sentences(narration, limit=1)
    words = re.findall(r"[A-Za-z']+", sentence)
    if not words:
        return normalize_title("Quick Learning Boost")
    title = " ".join(words[:6]).title()
    return normalize_title(title)


def build_hashtags(article: Article, narration: str) -> str:
    text = f"{article.title} {article.body} {narration}".lower()
    hashtags: list[str] = ["#Education"]

    keyword_map = [
        ("#Productivity", ["productivity", "focus", "task", "work", "efficiency"]),
        ("#SelfImprovement", ["growth", "habit", "mindset", "improve", "goal"]),
        ("#Technology", ["technology", "ai", "code", "software", "automation", "python"]),
        ("#StudyTips", ["study", "learn", "education", "teacher", "skill"]),
        ("#Japan", ["japan", "japanese", "kyoto", "tokyo"]),
    ]

    for hashtag, keywords in keyword_map:
        if hashtag == "#Japan" and float(os.environ.get("JAPAN_STYLE_LEVEL", "0")) >= 0.3:
            hashtags.append(hashtag)
            continue
        if any(keyword in text for keyword in keywords):
            hashtags.append(hashtag)

    deduped: list[str] = []
    for hashtag in hashtags:
        if hashtag not in deduped:
            deduped.append(hashtag)
    return " ".join(deduped[:3])


def build_fallback_metadata(article: Article, narration: str) -> UploadMetadata:
    title = build_fallback_title(narration)
    summary = extract_first_sentences(narration, limit=2) or normalize_summary(narration[:220])
    hashtags = build_hashtags(article, narration)
    description = f"{article.link}\n\n{summary}\n\n{hashtags}".strip()
    return UploadMetadata(title=title, description=description)


def parse_model_metadata(raw_text: str) -> UploadMetadata | None:
    cleaned = cleanup_model_text(raw_text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    title = normalize_title(str(payload.get("title", "")))
    summary = normalize_summary(str(payload.get("summary", "")))
    hashtags = " ".join(str(payload.get("hashtags", "")).split()).strip()

    if not title or not summary:
        return None

    description = f"{payload.get('url', '')}\n\n{summary}".strip()
    if hashtags:
        description = f"{description}\n\n{hashtags}"
    return UploadMetadata(title=title, description=description)


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

    metadata_prompt = f"""
Create YouTube Shorts metadata for an English-speaking audience.

Return valid JSON only using this exact schema:
{{
  "title": "catchy title without #Shorts",
  "summary": "two short English sentences",
  "hashtags": "#TagOne #TagTwo #TagThree",
  "url": "{article.link}"
}}

Rules:
- Make the title catchy and click-worthy for global viewers
- Keep the title concise and natural
- Do not include #Shorts inside title
- Summary should be easy to scan and sound native in English
- Hashtags should be strategic and relevant
- Include 2 to 3 hashtags, and prefer English hashtags

Article title:
{article.title}

Article body:
{article.body[:3000]}

Narration:
{narration[:1200]}
""".strip()

    try:
        metadata_response = generate_with_retry(
            client=client,
            model_name=model_name,
            prompt=metadata_prompt,
            max_attempts=2,
        )
        parsed = parse_model_metadata(metadata_response.text or "")
        if parsed is not None:
            return parsed
        print("Gemini metadata response could not be parsed. Using fallback metadata.")
    except Exception as exc:
        print(f"Gemini metadata generation failed. Using fallback metadata. Error: {exc}")

    return build_fallback_metadata(article, narration)


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
