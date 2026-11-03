#!/usr/bin/env python
"""End-to-end flow: fetch article, generate video, upload to YouTube."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from video_generator import (
    Article,
    fetch_recent_articles,
    generate_video,
    get_client,
    resolve_source_url,
)
from youtube_uploader import upload_short


def load_post_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_post_log(log_path: Path, entries: list[dict]) -> None:
    log_path.write_text(
        json.dumps(entries, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def is_article_already_posted(entries: list[dict], article_link: str) -> bool:
    return any(entry.get("article_link") == article_link for entry in entries)


def parse_source_urls() -> list[str]:
    raw_urls = (
        os.getenv("SOURCE_URLS")
        or os.getenv("NOTE_URL")
        or os.getenv("RSS_SOURCE_URL")
        or ""
    )
    urls = [
        line.strip()
        for line in raw_urls.replace(",", "\n").splitlines()
        if line.strip()
    ]
    if not urls:
        return [resolve_source_url(None)]
    return list(dict.fromkeys(urls))


def choose_random_source(source_urls: list[str]) -> str:
    if not source_urls:
        raise ValueError("No source URLs were configured.")
    return random.choice(source_urls)


def choose_article_from_source(
    source_url: str,
    log_entries: list[dict],
    recent_limit: int,
) -> Article | None:
    articles = fetch_recent_articles(source_url, limit=recent_limit)
    for article in articles:
        if article.link and not is_article_already_posted(log_entries, article.link):
            return article
    return None


def cleanup_generated_files(paths: list[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            path.unlink()


def log_run_context() -> None:
    now_utc = datetime.now(timezone.utc)
    now_jst = now_utc.astimezone(ZoneInfo("Asia/Tokyo"))
    print("Run context:")
    print(f"- utc_now: {now_utc.isoformat()}")
    print(f"- jst_now: {now_jst.isoformat()}")
    print("- target_schedule: every 4 hours for automated runs")


def main() -> None:
    load_dotenv()
    log_run_context()

    log_path = Path(os.getenv("POSTED_LOG_PATH", "posted_articles.json"))
    log_entries = load_post_log(log_path)
    source_urls = parse_source_urls()
    selected_source = choose_random_source(source_urls)
    print(f"Selected source: {selected_source}")
    article = choose_article_from_source(
        source_url=selected_source,
        log_entries=log_entries,
        recent_limit=int(os.getenv("RECENT_ARTICLE_LIMIT", "5")),
    )

    if article is None:
        print("No new unposted articles were found in the selected source.")
        return

    client = get_client()
    video_result = generate_video(
        article,
        client=client,
        target_seconds=int(os.getenv("TARGET_SECONDS", "45")),
        text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"),
        image_model=os.getenv("GEMINI_IMAGE_MODEL", "imagen-4.0-fast-generate-001"),
        image_fallback_model=os.getenv(
            "GEMINI_INLINE_IMAGE_MODEL", "gemini-2.5-flash-image"
        ),
        background_mode=os.getenv("BACKGROUND_MODE", "local"),
        background_theme=os.getenv("BACKGROUND_THEME", "auto"),
        background_seed=os.getenv("BACKGROUND_SEED") or None,
        japan_style_level=float(os.getenv("JAPAN_STYLE_LEVEL", "0.45")),
        font_path=os.getenv("SHORTS_FONT_PATH"),
        voice_lang=os.getenv("VOICE_LANGUAGE", "en"),
        voice_tld=os.getenv("VOICE_TLD", "com"),
        bgm_mode=os.getenv("BGM_MODE", "auto"),
        bgm_library_dir=os.getenv("BGM_LIBRARY_DIR", "bgm"),
        bgm_path=os.getenv("BGM_PATH") or None,
        bgm_volume=float(os.getenv("BGM_VOLUME", "0.12")),
        bgm_fadein=float(os.getenv("BGM_FADEIN", "1.0")),
        bgm_fadeout=float(os.getenv("BGM_FADEOUT", "1.5")),
        subtitle_style=os.getenv("SUBTITLE_STYLE", "karaoke"),
        subtitle_max_words=int(os.getenv("SUBTITLE_MAX_WORDS", "6")),
        subtitle_max_words_per_line=int(
            os.getenv("SUBTITLE_MAX_WORDS_PER_LINE", "3")
        ),
        subtitle_highlight_color=os.getenv("SUBTITLE_HIGHLIGHT_COLOR", "#FFD54A"),
        subtitle_position=os.getenv("SUBTITLE_POSITION", "center"),
        subtitle_y=int(os.getenv("SUBTITLE_Y", "1180")),
        audio_output=os.getenv("AUDIO_OUTPUT", "narration.mp3"),
        image_output=os.getenv("IMAGE_OUTPUT", "background.png"),
        cropped_image_output=os.getenv("CROPPED_IMAGE_OUTPUT", "background_9x16.png"),
        script_output=os.getenv("SCRIPT_OUTPUT", "narration.txt"),
        output=os.getenv("VIDEO_OUTPUT", "output.mp4"),
    )

    upload_result = upload_short(
        video_path=video_result.output_path,
        article=article,
        narration=video_result.narration,
        client=client,
        client_secrets_path=os.getenv(
            "YOUTUBE_CLIENT_SECRETS", "client_secrets.json"
        ),
        token_path=os.getenv("YOUTUBE_TOKEN_FILE", "token.pickle"),
        title_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"),
        privacy_status=os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
    )

    log_entries.append(
        {
            "article_link": article.link,
            "article_title": article.title,
            "feed_url": article.feed_url,
            "video_id": upload_result.video_id,
            "youtube_title": upload_result.title,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_post_log(log_path, log_entries)

    cleanup_generated_files(
        [
            video_result.script_path,
            video_result.audio_path,
            video_result.image_path,
            video_result.cropped_image_path,
            video_result.output_path,
        ]
    )

    print("Workflow completed:")
    print(f"- article: {article.title}")
    print(f"- source: {article.feed_url}")
    print(f"- uploaded video_id: {upload_result.video_id}")
    print(f"- log: {log_path}")


if __name__ == "__main__":
    main()
