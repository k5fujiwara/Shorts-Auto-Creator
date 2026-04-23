#!/usr/bin/env python
"""Generate a YouTube Shorts video from the latest RSS article.

The Gemini image path uses the current `google-genai` SDK, which is the
maintained Python SDK for Gemini/Imagen image generation.

Example:
    python video_generator.py "https://note.com/example" --target-seconds 15
"""

from __future__ import annotations

import argparse
import hashlib
import html
import math
import os
import random
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import feedparser
import numpy as np
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from dotenv import load_dotenv

try:
    from moviepy.editor import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip
except ImportError:
    from moviepy import AudioFileClip, CompositeAudioClip, CompositeVideoClip, ImageClip


VIDEO_SIZE = (1080, 1920)
SUBTITLE_CANVAS = (1000, 340)
SUBTITLE_Y = 1280
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "up",
    "with",
    "you",
    "your",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


@dataclass
class Article:
    title: str
    link: str
    published: str
    body: str
    feed_url: str


@dataclass
class SubtitleSegment:
    text: str
    start: float
    duration: float
    highlight_indices: tuple[int, ...] = ()


@dataclass
class VideoGenerationResult:
    article: Article
    narration: str
    script_path: Path
    audio_path: Path
    image_path: Path
    cropped_image_path: Path
    output_path: Path


def strip_html(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def build_feed_candidates(url: str) -> Iterable[str]:
    normalized = url.strip().rstrip("/")
    parsed = urlparse(normalized)
    yield normalized

    if "note.com" in parsed.netloc and not normalized.endswith("/rss"):
        yield f"{normalized}/rss"

    for suffix in ("/feed", "/rss", "/rss.xml", "/feed.xml", "/index.xml"):
        candidate = f"{normalized}{suffix}"
        if candidate != normalized:
            yield candidate


def discover_feed(url: str) -> tuple[str, feedparser.FeedParserDict]:
    tried: list[str] = []
    seen: set[str] = set()

    for candidate in build_feed_candidates(url):
        if candidate in seen:
            continue
        seen.add(candidate)
        feed = feedparser.parse(candidate)
        tried.append(candidate)
        if feed.entries:
            return candidate, feed

    raise RuntimeError(
        "RSS feed could not be found. Tried:\n- " + "\n- ".join(tried)
    )


def extract_article_text(entry: feedparser.FeedParserDict) -> str:
    parts: list[str] = []

    for key in ("summary", "description"):
        value = strip_html(entry.get(key, ""))
        if value:
            parts.append(value)

    for content_item in entry.get("content", []):
        value = strip_html(content_item.get("value", ""))
        if value:
            parts.append(value)

    deduped: list[str] = []
    for item in parts:
        if item and item not in deduped:
            deduped.append(item)

    return "\n\n".join(deduped).strip()


def convert_entry_to_article(
    feed_url: str,
    entry: feedparser.FeedParserDict,
) -> Article | None:
    title = entry.get("title", "Untitled article").strip()
    link = entry.get("link", "").strip()
    published = (entry.get("published") or entry.get("updated") or "").strip()
    body = extract_article_text(entry)

    if not body:
        return None

    return Article(
        title=title,
        link=link,
        published=published,
        body=body,
        feed_url=feed_url,
    )


def fetch_recent_articles(url: str, limit: int = 5) -> list[Article]:
    feed_url, feed = discover_feed(url)
    articles: list[Article] = []

    for entry in feed.entries[:limit]:
        article = convert_entry_to_article(feed_url, entry)
        if article is not None:
            articles.append(article)

    if not articles:
        raise RuntimeError("RSS items were found, but none had usable content.")

    return articles


def fetch_latest_article(url: str) -> Article:
    return fetch_recent_articles(url, limit=1)[0]


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY in your environment variables.")
    return genai.Client(api_key=api_key)


def resolve_source_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url

    env_url = os.environ.get("NOTE_URL") or os.environ.get("RSS_SOURCE_URL")
    if env_url:
        return env_url

    raise EnvironmentError(
        "Provide a URL as an argument or set NOTE_URL in your .env file."
    )


def cleanup_model_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("```", "").strip()
    cleaned = re.sub(r"^\s*台本[:：]\s*", "", cleaned)
    return cleaned


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).upper()
    return status_code in {429, 500, 502, 503, 504} or any(
        token in message
        for token in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "HIGH DEMAND",
            "RESOURCE_EXHAUSTED",
        )
    )


def generate_with_retry(
    client: genai.Client,
    model_name: str,
    prompt: str,
    max_attempts: int = 4,
    config: object | None = None,
):
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            if attempt >= max_attempts or not is_retryable_api_error(exc):
                raise
            wait_seconds = min(2 ** attempt, 20)
            print(
                f"Gemini request failed temporarily ({exc}). "
                f"Retrying in {wait_seconds} seconds... [{attempt}/{max_attempts}]"
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed without returning a response.")


def generate_narration(
    client: genai.Client,
    article: Article,
    target_seconds: int,
    model_name: str,
) -> str:
    target_words = max(35, int(target_seconds * 2.1))
    prompt = f"""
You are writing a voiceover script for a YouTube Short.
Read the article below and turn it into a motivational English narration.

Requirements:
- Output in English only
- Around {target_words} words for about {target_seconds} seconds
- Hook the viewer in the first sentence
- Use natural spoken English
- Use short, clear sentences that are easy to hear
- Use simple words and clean punctuation
- Keep it uplifting, practical, and action-oriented
- Output one paragraph only
- No bullet points, no emojis, no hashtags, no title

Article title:
{article.title}

Published:
{article.published or "Unknown"}

Article URL:
{article.link or "Unknown"}

Article body:
{article.body[:6000]}
""".strip()

    response = generate_with_retry(client=client, model_name=model_name, prompt=prompt)
    text = cleanup_model_text(response.text or "")
    if not text:
        raise RuntimeError("Gemini did not return narration text.")
    return text


def generate_background_prompt(
    article: Article,
    narration: str,
    japan_style_level: float,
) -> str:
    japan_style_instruction = ""
    if japan_style_level > 0:
        japan_style_instruction = (
            "Blend in subtle Japanese design sensibilities for a global audience. "
            "Use modern Japanese aesthetics such as refined minimalism, soft washi-like texture, "
            "clean balance, elegant lighting, and understated motifs inspired by shoji, paper, "
            "zen composition, city neon, or seasonal Japanese color harmony. "
            "Keep it tasteful and not stereotypical."
        )

    return textwrap.dedent(
        f"""
        Create a visually striking 9:16 background image for a YouTube Shorts video.
        Theme: self-improvement, motivation, positive action.
        Style: cinematic, modern, clean, high contrast, premium, emotionally uplifting.
        Important: no text, no letters, no captions, no logo, no watermark, no UI.
        Use a single cohesive scene that matches this article.
        {japan_style_instruction}

        Article title: {article.title}
        Article summary: {article.body[:1000]}
        Narration mood: {narration[:500]}
        """
    ).strip()


def get_visual_themes() -> dict[str, dict[str, object]]:
    return {
        "productivity": {
            "keywords": [
                "productivity",
                "work",
                "focus",
                "task",
                "routine",
                "business",
                "仕事",
                "効率",
                "集中",
                "段取り",
                "確認",
                "タスク",
                "メール",
                "自動化",
            ],
            "palette": ((9, 18, 44), (27, 63, 122), (86, 184, 255)),
            "accent": (255, 196, 88, 170),
            "style": "panels",
        },
        "growth": {
            "keywords": [
                "growth",
                "habit",
                "goal",
                "motivation",
                "improve",
                "mindset",
                "成長",
                "習慣",
                "目標",
                "改善",
                "やる気",
                "行動",
            ],
            "palette": ((16, 28, 30), (21, 92, 84), (102, 224, 177)),
            "accent": (255, 214, 120, 165),
            "style": "upward",
        },
        "learning": {
            "keywords": [
                "learn",
                "study",
                "skill",
                "education",
                "teacher",
                "講師",
                "学習",
                "勉強",
                "教育",
                "スキル",
            ],
            "palette": ((34, 18, 59), (76, 46, 130), (181, 152, 255)),
            "accent": (128, 222, 255, 150),
            "style": "cards",
        },
        "technology": {
            "keywords": [
                "technology",
                "ai",
                "code",
                "software",
                "system",
                "python",
                "開発",
                "技術",
                "生成",
                "コード",
                "プログラム",
            ],
            "palette": ((5, 15, 24), (11, 82, 115), (82, 238, 255)),
            "accent": (108, 255, 195, 155),
            "style": "grid",
        },
    }


def pick_visual_theme(
    article: Article,
    narration: str,
    requested_theme: str = "auto",
) -> dict[str, object]:
    text = f"{article.title} {article.body} {narration}".lower()
    themes = get_visual_themes()

    if requested_theme in themes:
        selected = dict(themes[requested_theme])
        selected["name"] = requested_theme
        return selected

    best_name = "productivity"
    best_score = -1
    for name, theme in themes.items():
        score = sum(text.count(keyword) for keyword in theme["keywords"])
        if score > best_score:
            best_name = name
            best_score = score

    selected = dict(themes[best_name])
    selected["name"] = best_name
    return selected


def blend_color(color_a: tuple[int, int, int], color_b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(
        int(color_a[i] * (1 - ratio) + color_b[i] * ratio) for i in range(3)
    )


def add_bottom_shadow(width: int, height: int) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = overlay.load()
    start_y = int(height * 0.48)
    for y in range(start_y, height):
        ratio = (y - start_y) / max(height - start_y, 1)
        alpha = int(30 + 160 * ratio)
        for x in range(width):
            pixels[x, y] = (0, 0, 0, alpha)
    return overlay


def get_background_rng(background_seed: str | None):
    if background_seed:
        return random.Random(background_seed)
    return random.SystemRandom()


def vary_color(
    color: tuple[int, int, int],
    rng,
    min_factor: float = 0.85,
    max_factor: float = 1.18,
) -> tuple[int, int, int]:
    factor = rng.uniform(min_factor, max_factor)
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def build_glow_positions(width: int, height: int, rng) -> list[tuple[int, int, int, int]]:
    positions: list[tuple[int, int, int, int]] = []
    for _ in range(rng.randint(3, 5)):
        positions.append(
            (
                rng.randint(int(width * 0.12), int(width * 0.88)),
                rng.randint(int(height * 0.12), int(height * 0.72)),
                rng.randint(int(width * 0.12), int(width * 0.34)),
                rng.randint(55, 130),
            )
        )
    return positions


def draw_random_light_streaks(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    rng,
) -> None:
    for _ in range(rng.randint(5, 9)):
        y = rng.randint(80, int(height * 0.72))
        angle_shift = rng.randint(-180, 180)
        thickness = rng.randint(3, 8)
        draw.line(
            (0, y, width, y + angle_shift),
            fill=(255, 255, 255, rng.randint(10, 28)),
            width=thickness,
        )


def draw_japanese_motifs(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    rng,
    accent: tuple[int, int, int, int],
    japan_style_level: float,
) -> None:
    motif_strength = max(0.0, min(1.0, japan_style_level))
    if motif_strength <= 0:
        return

    motif_alpha = int(28 + 70 * motif_strength)
    motif_color = (accent[0], accent[1], accent[2], motif_alpha)

    for _ in range(rng.randint(3, 6)):
        y = rng.randint(120, int(height * 0.62))
        draw.line(
            (80, y, width - 80, y),
            fill=(255, 255, 255, int(10 + 26 * motif_strength)),
            width=rng.randint(1, 3),
        )

    for _ in range(rng.randint(2, 4)):
        x = rng.randint(120, width - 120)
        draw.line(
            (x, 120, x, int(height * 0.72)),
            fill=(255, 255, 255, int(8 + 22 * motif_strength)),
            width=rng.randint(1, 2),
        )

    for _ in range(rng.randint(2, 4)):
        cx = rng.randint(int(width * 0.18), int(width * 0.82))
        cy = rng.randint(int(height * 0.16), int(height * 0.56))
        radius = rng.randint(30, 90)
        draw.arc(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            start=rng.randint(180, 240),
            end=rng.randint(300, 360),
            fill=motif_color,
            width=rng.randint(3, 6),
        )


def resolve_subtitle_position(
    subtitle_position: str,
    subtitle_y: int,
) -> tuple[str, str | int]:
    if subtitle_position == "center":
        return ("center", "center")
    return ("center", subtitle_y)


def list_audio_files(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS]
    )


def get_bgm_category_preferences(
    theme_name: str,
    japan_style_level: float,
) -> list[str]:
    category_map = {
        "productivity": ["bright", "tech", "japanese", "calm"],
        "growth": ["bright", "calm", "japanese", "tech"],
        "learning": ["calm", "japanese", "bright", "tech"],
        "technology": ["tech", "bright", "japanese", "calm"],
    }
    preferences = list(category_map.get(theme_name, ["bright", "calm", "tech", "japanese"]))
    if japan_style_level >= 0.35 and "japanese" in preferences:
        preferences.remove("japanese")
        insert_at = 1 if preferences else 0
        preferences.insert(insert_at, "japanese")
    return preferences


def choose_bgm_path(
    *,
    article: Article,
    narration: str,
    bgm_mode: str,
    manual_bgm_path: str | None,
    bgm_library_dir: str,
    background_theme: str,
    japan_style_level: float,
    selection_seed: str | None,
) -> str | None:
    if manual_bgm_path:
        return manual_bgm_path

    if bgm_mode != "auto":
        return None

    library_dir = Path(bgm_library_dir)
    rng = get_background_rng(selection_seed)
    theme = pick_visual_theme(article, narration, requested_theme=background_theme)
    preferences = get_bgm_category_preferences(theme["name"], japan_style_level)

    for category in preferences:
        files = list_audio_files(library_dir / category)
        if files:
            chosen = rng.choice(files)
            print(f"Selected BGM category: {category}")
            print(f"Selected BGM file: {chosen}")
            return str(chosen)

    fallback_files = list_audio_files(library_dir)
    if fallback_files:
        chosen = rng.choice(fallback_files)
        print(f"Selected fallback BGM file: {chosen}")
        return str(chosen)

    print(f"No usable BGM files were found under: {library_dir}")
    return None


def generate_local_fallback_image(
    article: Article,
    narration: str,
    output_path: Path,
    background_theme: str,
    background_seed: str | None,
    japan_style_level: float,
) -> None:
    seed_source = f"{article.title}|{article.link}|{article.published}"
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    rng = get_background_rng(background_seed)
    theme = pick_visual_theme(article, narration, requested_theme=background_theme)
    palette = theme["palette"]
    color_top = vary_color(palette[0], rng, 0.82, 1.05)
    color_mid = vary_color(palette[1], rng, 0.88, 1.12)
    color_glow = vary_color(palette[2], rng, 0.95, 1.25)
    accent = theme["accent"]
    style = theme["style"]

    width, height = VIDEO_SIZE
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        ratio = y / max(height - 1, 1)
        base = blend_color(color_top, color_mid, ratio)
        for x in range(width):
            side_glow = ((x / max(width - 1, 1)) - 0.5) ** 2
            lift = max(0, 1.0 - side_glow * 3.2)
            pixels[x, y] = tuple(
                min(255, int(base[i] + color_glow[i] * 0.12 * lift)) for i in range(3)
            )

    base_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    glow_positions = build_glow_positions(width, height, rng)
    for cx, cy, radius, alpha_scale in glow_positions:
        glow_fill = (color_glow[0], color_glow[1], color_glow[2], alpha_scale)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=glow_fill)

    if style == "panels":
        panel_count = rng.randint(3, 5)
        for idx in range(panel_count):
            top = 160 + idx * rng.randint(140, 185)
            left = rng.randint(80, 170) + idx * rng.randint(18, 36)
            right = width - rng.randint(90, 170) - idx * rng.randint(0, 12)
            bottom = top + rng.randint(90, 130)
            draw.rounded_rectangle(
                (left, top, right, bottom),
                radius=28,
                fill=(255, 255, 255, 22),
                outline=(255, 255, 255, 42),
                width=2,
            )
        for idx, bar_height in enumerate(
            sorted([rng.randint(150, 420) for _ in range(3)])
        ):
            x0 = 120 + idx * rng.randint(95, 140)
            draw.rounded_rectangle(
                (x0, 980 - bar_height, x0 + 64, 980),
                radius=18,
                fill=(accent[0], accent[1], accent[2], 125),
            )
    elif style == "upward":
        path_points = [
            (rng.randint(90, 180), rng.randint(1180, 1360)),
            (rng.randint(220, 330), rng.randint(980, 1140)),
            (rng.randint(380, 500), rng.randint(860, 1020)),
            (rng.randint(560, 690), rng.randint(700, 860)),
            (rng.randint(720, 860), rng.randint(520, 700)),
            (rng.randint(860, 980), rng.randint(320, 480)),
        ]
        for idx in range(len(path_points) - 1):
            draw.line(
                (path_points[idx], path_points[idx + 1]),
                fill=(255, 255, 255, 110),
                width=rng.randint(16, 24),
            )
        arrow_tip_x, arrow_tip_y = path_points[-1]
        draw.polygon(
            [
                (arrow_tip_x - 40, arrow_tip_y + 20),
                (arrow_tip_x + 42, arrow_tip_y + 6),
                (arrow_tip_x, arrow_tip_y - 68),
            ],
            fill=(accent[0], accent[1], accent[2], 180),
        )
    elif style == "cards":
        card_boxes = []
        for idx in range(3):
            left = 90 + idx * rng.randint(45, 85)
            top = 190 + idx * rng.randint(110, 180)
            right = width - rng.randint(90, 220)
            bottom = top + rng.randint(260, 380)
            card_boxes.append((left, top, right, bottom))
        for box in card_boxes:
            draw.rounded_rectangle(
                box,
                radius=36,
                fill=(255, 255, 255, 28),
                outline=(255, 255, 255, 58),
                width=3,
            )
    elif style == "grid":
        x_spacing = rng.randint(95, 135)
        y_spacing = rng.randint(95, 135)
        for x in range(80, width, x_spacing):
            draw.line((x, 140, x, 980), fill=(255, 255, 255, 28), width=2)
        for y in range(160, 1020, y_spacing):
            draw.line((60, y, width - 60, y), fill=(255, 255, 255, 24), width=2)
        nodes = [
            (rng.randint(130, 250), rng.randint(220, 330)),
            (rng.randint(300, 450), rng.randint(360, 500)),
            (rng.randint(560, 720), rng.randint(300, 430)),
            (rng.randint(760, 930), rng.randint(520, 680)),
            (rng.randint(430, 620), rng.randint(720, 880)),
        ]
        for cx, cy in nodes:
            radius = rng.randint(14, 22)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=accent)
        for start, end in zip(nodes, nodes[1:]):
            draw.line((start, end), fill=(accent[0], accent[1], accent[2], 100), width=6)

    draw_random_light_streaks(draw, width, height, rng)
    draw_japanese_motifs(
        draw,
        width,
        height,
        rng,
        accent,
        japan_style_level=japan_style_level,
    )

    final_image = Image.alpha_composite(base_rgba, overlay)
    final_image = Image.alpha_composite(final_image, add_bottom_shadow(width, height))
    final_image.convert("RGB").save(output_path)


def generate_theme_image(
    client: genai.Client,
    article: Article,
    narration: str,
    output_path: Path,
    image_model: str,
    fallback_image_model: str,
    background_mode: str,
    background_theme: str,
    background_seed: str | None,
    japan_style_level: float,
) -> None:
    if background_mode == "local":
        print("Using local background generation to stay within the free tier.")
        generate_local_fallback_image(
            article,
            narration,
            output_path,
            background_theme=background_theme,
            background_seed=background_seed,
            japan_style_level=japan_style_level,
        )
        return

    prompt = generate_background_prompt(article, narration, japan_style_level)
    errors: list[str] = []

    try:
        response = client.models.generate_images(
            model=image_model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                output_mime_type="image/png",
            ),
        )
        response.generated_images[0].image.save(output_path)
        return
    except Exception as exc:  # pragma: no cover - SDK/runtime dependent
        errors.append(f"{image_model}: {exc}")

    try:
        response = generate_with_retry(
            client=client,
            model_name=fallback_image_model,
            prompt=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="9:16"),
            ),
        )
        for part in getattr(response, "parts", []) or []:
            if getattr(part, "inline_data", None):
                part.as_image().save(output_path)
                return
        errors.append(f"{fallback_image_model}: no image part returned")
    except Exception as exc:  # pragma: no cover - SDK/runtime dependent
        errors.append(f"{fallback_image_model}: {exc}")

    print("Image generation via Gemini failed. Falling back to local background.")
    for error in errors:
        print(f"- {error}")
    generate_local_fallback_image(
        article,
        narration,
        output_path,
        background_theme=background_theme,
        background_seed=background_seed,
        japan_style_level=japan_style_level,
    )


def synthesize_mp3(text: str, output_path: Path, voice_lang: str, voice_tld: str) -> None:
    tts = gTTS(text=text, lang=voice_lang, tld=voice_tld)
    tts.save(str(output_path))


def detect_font_path(user_font_path: str | None) -> str:
    if user_font_path and Path(user_font_path).exists():
        return user_font_path

    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\ARLRDBD.TTF",
        r"C:\Windows\Fonts\seguiemj.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\verdanab.ttf",
        r"C:\Windows\Fonts\trebucbd.ttf",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\BIZ-UDGothicR.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Suitable subtitle font not found. Pass --font-path or set SHORTS_FONT_PATH."
    )


def prepare_background_image(source_path: Path, output_path: Path) -> None:
    target_w, target_h = VIDEO_SIZE

    with Image.open(source_path).convert("RGB") as image:
        scale = max(target_w / image.width, target_h / image.height)
        resized = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS,
        )

        left = (resized.width - target_w) // 2
        top = (resized.height - target_h) // 2
        cropped = resized.crop((left, top, left + target_w, top + target_h))
        cropped.save(output_path)


def split_subtitle_text(text: str, max_words: int = 6) -> list[str]:
    normalized = SPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[,;:])\s+", normalized)
    chunks: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue
        for index in range(0, len(words), max_words):
            piece = " ".join(words[index : index + max_words]).strip()
            if piece:
                chunks.append(piece)

    merged: list[str] = []
    for chunk in chunks or [normalized]:
        if merged and len(chunk.split()) <= 2:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged or [normalized]


def wrap_subtitle_text(text: str, max_words_per_line: int = 3) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = current + [word]
        if current and (
            len(candidate) > max_words_per_line or len(" ".join(candidate)) > 18
        ):
            lines.append(" ".join(current))
            current = [word]
        else:
            current = candidate

    if current:
        lines.append(" ".join(current))

    if len(lines) > 3:
        lines = lines[:2] + [" ".join(" ".join(lines[2:]).split())]

    return "\n".join(lines)


def split_words_for_render(text: str, max_words_per_line: int = 3) -> list[list[tuple[int, str]]]:
    words = text.split()
    lines: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []

    for index, word in enumerate(words):
        candidate = current + [(index, word)]
        candidate_text = " ".join(token for _, token in candidate)
        if current and (
            len(candidate) > max_words_per_line or len(candidate_text) > 18
        ):
            lines.append(current)
            current = [(index, word)]
        else:
            current = candidate

    if current:
        lines.append(current)

    if len(lines) > 3:
        flattened = [item for line in lines for item in line]
        lines = [flattened[:max_words_per_line], flattened[max_words_per_line : max_words_per_line * 2], flattened[max_words_per_line * 2 :]]
        lines = [line for line in lines if line]

    return lines


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-zA-Z']", "", word).lower()


def pick_keyword_indices(text: str) -> tuple[int, ...]:
    words = text.split()
    scored: list[tuple[tuple[int, int], int]] = []

    for index, word in enumerate(words):
        normalized = normalize_word(word)
        if len(normalized) < 4 or normalized in ENGLISH_STOP_WORDS:
            continue
        score = (len(normalized), 1 if word[:1].isupper() else 0)
        scored.append((score, index))

    scored.sort(reverse=True)
    selected = [index for _, index in scored[:2]]
    return tuple(sorted(selected))


def parse_hex_color(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        return fallback
    try:
        return tuple(int(cleaned[i : i + 2], 16) for i in range(0, 6, 2))
    except ValueError:
        return fallback


def render_subtitle_image(
    text: str,
    font_path: str,
    highlight_indices: tuple[int, ...] = (),
    highlight_color: tuple[int, int, int] = (255, 213, 74),
    max_words_per_line: int = 3,
    font_size: int = 78,
    stroke_width: int = 7,
) -> np.ndarray:
    canvas = Image.new("RGBA", SUBTITLE_CANVAS, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, font_size)
    lines = split_words_for_render(text, max_words_per_line=max_words_per_line)
    line_spacing = 14
    space_bbox = draw.textbbox((0, 0), " ", font=font, stroke_width=stroke_width)
    space_width = space_bbox[2] - space_bbox[0]

    line_metrics: list[tuple[list[tuple[int, str]], int, int]] = []
    max_width = 0
    total_height = 0

    for line in lines:
        width = 0
        max_height = 0
        for word_index, word in line:
            bbox = draw.textbbox((0, 0), word, font=font, stroke_width=stroke_width)
            word_width = bbox[2] - bbox[0]
            word_height = bbox[3] - bbox[1]
            width += word_width
            max_height = max(max_height, word_height)
            if word_index != line[-1][0]:
                width += space_width
        line_metrics.append((line, width, max_height))
        max_width = max(max_width, width)
        total_height += max_height

    if line_metrics:
        total_height += line_spacing * (len(line_metrics) - 1)

    text_w = max_width
    text_h = total_height
    x = (canvas.width - text_w) / 2
    y = (canvas.height - text_h) / 2

    padding_x = 44
    padding_y = 26
    draw.rounded_rectangle(
        (
            x - padding_x,
            y - padding_y,
            x + text_w + padding_x,
            y + text_h + padding_y,
        ),
        radius=34,
        fill=(8, 12, 20, 176),
    )
    shadow_offset = 4
    cursor_y = y

    for line, line_width, line_height in line_metrics:
        cursor_x = (canvas.width - line_width) / 2
        for word_index, word in line:
            bbox = draw.textbbox((0, 0), word, font=font, stroke_width=stroke_width)
            word_width = bbox[2] - bbox[0]
            fill_color = (
                (*highlight_color, 255)
                if word_index in highlight_indices
                else (248, 250, 255, 255)
            )
            draw.text(
                (cursor_x + shadow_offset, cursor_y + shadow_offset),
                word,
                font=font,
                fill=(0, 0, 0, 180),
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0, 210),
            )
            draw.text(
                (cursor_x, cursor_y),
                word,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width,
                stroke_fill=(10, 12, 16, 255),
            )
            cursor_x += word_width
            if word_index != line[-1][0]:
                cursor_x += space_width
        cursor_y += line_height + line_spacing
    return np.array(canvas)


def build_subtitle_segments(
    text: str,
    total_duration: float,
    subtitle_style: str,
    subtitle_max_words: int,
) -> list[SubtitleSegment]:
    chunks = split_subtitle_text(text, max_words=subtitle_max_words)
    weights = [max(len(chunk.replace(" ", "")), 1) for chunk in chunks]
    total_weight = sum(weights)
    current_time = 0.0
    segments: list[SubtitleSegment] = []

    for index, chunk in enumerate(chunks):
        is_last = index == len(chunks) - 1
        duration = total_duration * (weights[index] / total_weight)
        start = current_time
        end = total_duration if is_last else min(total_duration, current_time + duration)
        chunk_duration = max(end - start, 0.8)

        if subtitle_style == "karaoke":
            words = chunk.split()
            if len(words) <= 1:
                segments.append(
                    SubtitleSegment(
                        text=chunk,
                        start=start,
                        duration=chunk_duration,
                        highlight_indices=(0,) if words else (),
                    )
                )
            else:
                word_weights = [max(len(normalize_word(word)), 1) for word in words]
                total_word_weight = sum(word_weights)
                chunk_cursor = start
                for word_idx, word_weight in enumerate(word_weights):
                    word_duration = chunk_duration * (word_weight / total_word_weight)
                    word_end = end if word_idx == len(word_weights) - 1 else min(end, chunk_cursor + word_duration)
                    segments.append(
                        SubtitleSegment(
                            text=chunk,
                            start=chunk_cursor,
                            duration=max(word_end - chunk_cursor, 0.18),
                            highlight_indices=(word_idx,),
                        )
                    )
                    chunk_cursor = word_end
        else:
            highlight_indices = pick_keyword_indices(chunk) if subtitle_style == "keywords" else ()
            segments.append(
                SubtitleSegment(
                    text=chunk,
                    start=start,
                    duration=chunk_duration,
                    highlight_indices=highlight_indices,
                )
            )
        current_time = end

    return segments


def apply_clip_method(clip, modern_name: str, legacy_name: str, *args):
    modern_method = getattr(clip, modern_name, None)
    if callable(modern_method):
        return modern_method(*args)

    legacy_method = getattr(clip, legacy_name, None)
    if callable(legacy_method):
        return legacy_method(*args)

    raise AttributeError(
        f"Clip does not support '{modern_name}' or '{legacy_name}'."
    )


def set_clip_volume(clip, volume: float):
    modern_method = getattr(clip, "with_volume_scaled", None)
    if callable(modern_method):
        return modern_method(volume)

    legacy_method = getattr(clip, "volumex", None)
    if callable(legacy_method):
        return legacy_method(volume)

    return clip


def compute_fade_multiplier(
    t,
    duration: float,
    fade_in: float,
    fade_out: float,
):
    time_array = np.asarray(t, dtype=float)
    multiplier = np.ones_like(time_array, dtype=float)

    if fade_in > 0:
        multiplier = np.minimum(multiplier, np.clip(time_array / fade_in, 0.0, 1.0))

    if fade_out > 0:
        remaining = np.clip((duration - time_array) / fade_out, 0.0, 1.0)
        multiplier = np.minimum(multiplier, remaining)

    if np.isscalar(t):
        return float(multiplier)
    return multiplier


def scale_audio_frame(frame, multiplier):
    frame_array = np.asarray(frame)
    multiplier_array = np.asarray(multiplier)

    while multiplier_array.ndim < frame_array.ndim:
        multiplier_array = np.expand_dims(multiplier_array, axis=-1)

    return frame_array * multiplier_array


def apply_audio_fades(clip, fade_in: float, fade_out: float, duration: float):
    if fade_in <= 0 and fade_out <= 0:
        return clip

    def transform_frame(get_frame, t):
        frame = get_frame(t)
        multiplier = compute_fade_multiplier(
            t,
            duration=duration,
            fade_in=fade_in,
            fade_out=fade_out,
        )
        return scale_audio_frame(frame, multiplier)

    modern_method = getattr(clip, "transform", None)
    if callable(modern_method):
        return modern_method(transform_frame)

    legacy_method = getattr(clip, "fl", None)
    if callable(legacy_method):
        return legacy_method(transform_frame)

    return clip


def build_bgm_clip(
    bgm_path: str | None,
    target_duration: float,
    bgm_volume: float,
    bgm_fadein: float,
    bgm_fadeout: float,
) -> tuple[AudioFileClip | None, list[AudioFileClip]]:
    if not bgm_path:
        return None, []

    bgm_file = Path(bgm_path)
    if not bgm_file.exists():
        print(f"BGM file was not found and will be skipped: {bgm_path}")
        return None, []

    base_clip = AudioFileClip(str(bgm_file))
    if base_clip.duration <= 0:
        base_clip.close()
        return None, []

    loop_count = max(1, math.ceil(target_duration / base_clip.duration))
    bgm_segments: list[AudioFileClip] = []

    try:
        for index in range(loop_count):
            segment = AudioFileClip(str(bgm_file))
            start_time = index * base_clip.duration
            remaining = max(target_duration - start_time, 0)
            if remaining <= 0:
                segment.close()
                break
            if remaining < segment.duration:
                segment = apply_clip_method(
                    segment,
                    "subclipped",
                    "subclip",
                    0,
                    remaining,
                )
            segment = apply_clip_method(
                segment,
                "with_start",
                "set_start",
                start_time,
            )
            segment = set_clip_volume(segment, bgm_volume)
            bgm_segments.append(segment)

        if not bgm_segments:
            return None, [base_clip]

        bgm_mix = CompositeAudioClip(bgm_segments)
        bgm_mix = apply_clip_method(
            bgm_mix,
            "with_duration",
            "set_duration",
            target_duration,
        )
        bgm_mix = apply_audio_fades(
            bgm_mix,
            fade_in=bgm_fadein,
            fade_out=bgm_fadeout,
            duration=target_duration,
        )
        return bgm_mix, [base_clip, *bgm_segments]
    except Exception:
        base_clip.close()
        for segment in bgm_segments:
            try:
                segment.close()
            except Exception:
                pass
        raise


def compose_video(
    background_path: Path,
    audio_path: Path,
    narration: str,
    font_path: str,
    output_path: Path,
    bgm_path: str | None,
    bgm_volume: float,
    bgm_fadein: float,
    bgm_fadeout: float,
    subtitle_style: str,
    subtitle_max_words: int,
    subtitle_max_words_per_line: int,
    subtitle_highlight_color: tuple[int, int, int],
    subtitle_position: str,
    subtitle_y: int,
) -> None:
    audio_clip = AudioFileClip(str(audio_path))
    managed_audio_clips: list[object] = [audio_clip]
    background_clip = apply_clip_method(
        ImageClip(str(background_path)),
        "with_duration",
        "set_duration",
        audio_clip.duration,
    )
    clips = [background_clip]
    subtitle_anchor = resolve_subtitle_position(subtitle_position, subtitle_y)

    for segment in build_subtitle_segments(
        narration,
        audio_clip.duration,
        subtitle_style=subtitle_style,
        subtitle_max_words=subtitle_max_words,
    ):
        subtitle_image = render_subtitle_image(
            segment.text,
            font_path,
            highlight_indices=segment.highlight_indices,
            highlight_color=subtitle_highlight_color,
            max_words_per_line=subtitle_max_words_per_line,
        )
        subtitle_clip = apply_clip_method(
            apply_clip_method(
                apply_clip_method(
                    ImageClip(subtitle_image),
                    "with_start",
                    "set_start",
                    segment.start,
                ),
                "with_duration",
                "set_duration",
                segment.duration,
            ),
            "with_position",
            "set_position",
            subtitle_anchor,
        )
        clips.append(subtitle_clip)

    final_audio = audio_clip
    bgm_clip, bgm_sources = build_bgm_clip(
        bgm_path=bgm_path,
        target_duration=audio_clip.duration,
        bgm_volume=bgm_volume,
        bgm_fadein=bgm_fadein,
        bgm_fadeout=bgm_fadeout,
    )
    managed_audio_clips.extend(bgm_sources)
    if bgm_clip is not None:
        final_audio = CompositeAudioClip([audio_clip, bgm_clip])
        final_audio = apply_clip_method(
            final_audio,
            "with_duration",
            "set_duration",
            audio_clip.duration,
        )
        managed_audio_clips.append(final_audio)

    video = apply_clip_method(
        CompositeVideoClip(clips, size=VIDEO_SIZE),
        "with_duration",
        "set_duration",
        audio_clip.duration,
    )
    video = apply_clip_method(video, "with_audio", "set_audio", final_audio)
    video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
    )

    video.close()
    background_clip.close()
    for clip in reversed(managed_audio_clips):
        try:
            clip.close()
        except Exception:
            pass


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 9:16 YouTube Shorts video from the latest RSS article."
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="RSS URL or site URL. If omitted, NOTE_URL in .env is used.",
    )
    parser.add_argument(
        "--target-seconds",
        type=int,
        default=45,
        help="Narration target length. Use 15 first for faster iteration.",
    )
    parser.add_argument(
        "--text-model",
        default=os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash-lite"),
    )
    parser.add_argument(
        "--background-mode",
        choices=("local", "gemini"),
        default=os.environ.get("BACKGROUND_MODE", "local"),
        help="Use 'local' to stay within the free tier, or 'gemini' to try AI image generation.",
    )
    parser.add_argument(
        "--background-theme",
        choices=("auto", "productivity", "growth", "learning", "technology"),
        default=os.environ.get("BACKGROUND_THEME", "auto"),
        help="Visual theme for local backgrounds.",
    )
    parser.add_argument(
        "--background-seed",
        default=os.environ.get("BACKGROUND_SEED") or None,
        help="Optional seed to reproduce a specific local background.",
    )
    parser.add_argument(
        "--japan-style-level",
        type=float,
        default=float(os.environ.get("JAPAN_STYLE_LEVEL", "0.45")),
        help="How strongly to blend Japanese visual taste into backgrounds.",
    )
    parser.add_argument(
        "--image-model",
        default=os.environ.get("GEMINI_IMAGE_MODEL", "imagen-4.0-fast-generate-001"),
    )
    parser.add_argument(
        "--image-fallback-model",
        default=os.environ.get("GEMINI_INLINE_IMAGE_MODEL", "gemini-2.5-flash-image"),
    )
    parser.add_argument("--font-path", default=os.environ.get("SHORTS_FONT_PATH"))
    parser.add_argument(
        "--voice-lang",
        default=os.environ.get("VOICE_LANGUAGE", "en"),
        help="gTTS language code. Default is English.",
    )
    parser.add_argument(
        "--voice-tld",
        default=os.environ.get("VOICE_TLD", "com"),
        help="gTTS top-level domain, such as com or co.uk.",
    )
    parser.add_argument(
        "--bgm-path",
        default=os.environ.get("BGM_PATH"),
        help="Optional manual BGM file path.",
    )
    parser.add_argument(
        "--bgm-mode",
        choices=("auto", "manual", "off"),
        default=os.environ.get("BGM_MODE", "auto"),
        help="How BGM should be selected.",
    )
    parser.add_argument(
        "--bgm-library-dir",
        default=os.environ.get("BGM_LIBRARY_DIR", "bgm"),
        help="Folder that contains categorized BGM files.",
    )
    parser.add_argument(
        "--bgm-volume",
        type=float,
        default=float(os.environ.get("BGM_VOLUME", "0.12")),
        help="BGM volume relative to narration.",
    )
    parser.add_argument(
        "--bgm-fadein",
        type=float,
        default=float(os.environ.get("BGM_FADEIN", "1.0")),
        help="BGM fade-in duration in seconds.",
    )
    parser.add_argument(
        "--bgm-fadeout",
        type=float,
        default=float(os.environ.get("BGM_FADEOUT", "1.5")),
        help="BGM fade-out duration in seconds.",
    )
    parser.add_argument(
        "--subtitle-style",
        choices=("plain", "keywords", "karaoke"),
        default=os.environ.get("SUBTITLE_STYLE", "karaoke"),
        help="Subtitle emphasis style.",
    )
    parser.add_argument(
        "--subtitle-max-words",
        type=int,
        default=int(os.environ.get("SUBTITLE_MAX_WORDS", "6")),
        help="Maximum words per subtitle phrase.",
    )
    parser.add_argument(
        "--subtitle-max-words-per-line",
        type=int,
        default=int(os.environ.get("SUBTITLE_MAX_WORDS_PER_LINE", "3")),
        help="Maximum words per subtitle line.",
    )
    parser.add_argument(
        "--subtitle-highlight-color",
        default=os.environ.get("SUBTITLE_HIGHLIGHT_COLOR", "#FFD54A"),
        help="Highlight color in hex, e.g. #FFD54A.",
    )
    parser.add_argument(
        "--subtitle-position",
        choices=("center", "lower_third"),
        default=os.environ.get("SUBTITLE_POSITION", "center"),
        help="Subtitle placement preset.",
    )
    parser.add_argument(
        "--subtitle-y",
        type=int,
        default=int(os.environ.get("SUBTITLE_Y", str(SUBTITLE_Y))),
        help="Vertical subtitle position in pixels.",
    )
    parser.add_argument("--audio-output", default="narration.mp3")
    parser.add_argument("--image-output", default="background.png")
    parser.add_argument("--cropped-image-output", default="background_9x16.png")
    parser.add_argument("--script-output", default="narration.txt")
    parser.add_argument("--output", default="output.mp4")
    return parser.parse_args()


def generate_video(
    article: Article,
    *,
    client: genai.Client | None = None,
    target_seconds: int = 45,
    text_model: str = "gemini-2.5-flash-lite",
    image_model: str = "imagen-4.0-fast-generate-001",
    image_fallback_model: str = "gemini-2.5-flash-image",
    background_mode: str = "local",
    background_theme: str = "auto",
    background_seed: str | None = None,
    japan_style_level: float = 0.45,
    font_path: str | None = None,
    voice_lang: str = "en",
    voice_tld: str = "com",
    bgm_mode: str = "auto",
    bgm_library_dir: str = "bgm",
    bgm_path: str | None = None,
    bgm_volume: float = 0.12,
    bgm_fadein: float = 1.0,
    bgm_fadeout: float = 1.5,
    subtitle_style: str = "karaoke",
    subtitle_max_words: int = 6,
    subtitle_max_words_per_line: int = 3,
    subtitle_highlight_color: str = "#FFD54A",
    subtitle_position: str = "center",
    subtitle_y: int = SUBTITLE_Y,
    audio_output: str = "narration.mp3",
    image_output: str = "background.png",
    cropped_image_output: str = "background_9x16.png",
    script_output: str = "narration.txt",
    output: str = "output.mp4",
) -> VideoGenerationResult:
    client = client or get_client()

    audio_path = Path(audio_output)
    image_path = Path(image_output)
    cropped_image_path = Path(cropped_image_output)
    script_path = Path(script_output)
    output_path = Path(output)

    for path in (audio_path, image_path, cropped_image_path, script_path, output_path):
        ensure_parent(path)

    narration = generate_narration(
        client=client,
        article=article,
        target_seconds=target_seconds,
        model_name=text_model,
    )
    script_path.write_text(narration, encoding="utf-8")

    synthesize_mp3(
        narration,
        audio_path,
        voice_lang=voice_lang,
        voice_tld=voice_tld,
    )
    resolved_bgm_path = choose_bgm_path(
        article=article,
        narration=narration,
        bgm_mode=bgm_mode,
        manual_bgm_path=bgm_path,
        bgm_library_dir=bgm_library_dir,
        background_theme=background_theme,
        japan_style_level=japan_style_level,
        selection_seed=background_seed,
    )
    generate_theme_image(
        client=client,
        article=article,
        narration=narration,
        output_path=image_path,
        image_model=image_model,
        fallback_image_model=image_fallback_model,
        background_mode=background_mode,
        background_theme=background_theme,
        background_seed=background_seed,
        japan_style_level=japan_style_level,
    )
    prepare_background_image(image_path, cropped_image_path)
    resolved_font_path = detect_font_path(font_path)
    compose_video(
        background_path=cropped_image_path,
        audio_path=audio_path,
        narration=narration,
        font_path=resolved_font_path,
        output_path=output_path,
        bgm_path=resolved_bgm_path,
        bgm_volume=bgm_volume,
        bgm_fadein=bgm_fadein,
        bgm_fadeout=bgm_fadeout,
        subtitle_style=subtitle_style,
        subtitle_max_words=subtitle_max_words,
        subtitle_max_words_per_line=subtitle_max_words_per_line,
        subtitle_highlight_color=parse_hex_color(
            subtitle_highlight_color, (255, 213, 74)
        ),
        subtitle_position=subtitle_position,
        subtitle_y=subtitle_y,
    )

    return VideoGenerationResult(
        article=article,
        narration=narration,
        script_path=script_path,
        audio_path=audio_path,
        image_path=image_path,
        cropped_image_path=cropped_image_path,
        output_path=output_path,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_url = resolve_source_url(args.url)
    article = fetch_latest_article(source_url)
    result = generate_video(
        article,
        client=get_client(),
        target_seconds=args.target_seconds,
        text_model=args.text_model,
        image_model=args.image_model,
        image_fallback_model=args.image_fallback_model,
        background_mode=args.background_mode,
        background_theme=args.background_theme,
        background_seed=args.background_seed,
        japan_style_level=args.japan_style_level,
        font_path=args.font_path,
        voice_lang=args.voice_lang,
        voice_tld=args.voice_tld,
        bgm_mode=args.bgm_mode,
        bgm_library_dir=args.bgm_library_dir,
        bgm_path=args.bgm_path,
        bgm_volume=args.bgm_volume,
        bgm_fadein=args.bgm_fadein,
        bgm_fadeout=args.bgm_fadeout,
        subtitle_style=args.subtitle_style,
        subtitle_max_words=args.subtitle_max_words,
        subtitle_max_words_per_line=args.subtitle_max_words_per_line,
        subtitle_highlight_color=args.subtitle_highlight_color,
        subtitle_position=args.subtitle_position,
        subtitle_y=args.subtitle_y,
        audio_output=args.audio_output,
        image_output=args.image_output,
        cropped_image_output=args.cropped_image_output,
        script_output=args.script_output,
        output=args.output,
    )

    print("Completed:")
    print(f"- feed: {article.feed_url}")
    print(f"- article: {article.title}")
    print(f"- script: {result.script_path}")
    print(f"- audio: {result.audio_path}")
    print(f"- image: {result.cropped_image_path}")
    print(f"- video: {result.output_path}")


if __name__ == "__main__":
    main()
