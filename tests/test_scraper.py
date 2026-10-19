"""Unit tests for crawler helpers (robots, sitemap, config)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ollama_rag.scraper import (
    _canonicalize,
    _parse_sitemap_urls,
    _robots_allowed,
    scrape_website,
)


def test_canonicalize_strips_tracking():
    url = "https://WWW.Example.com/path/?utm_source=x&id=1#frag"
    assert _canonicalize(url) == "https://example.com/path?id=1"


def test_parse_sitemap():
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org and/schemas/sitemap/0.9">
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
      <url><loc>https://other.com/c</loc></url>
    </urlset>
    """.replace(" and/", "/")
    # Fix namespace - write properly
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
      <url><loc>https://other.com/c</loc></url>
    </urlset>
    """
    urls = _parse_sitemap_urls(xml, "example.com", limit=10)
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
    assert all("other.com" not in u for u in urls)


def test_robots_allowed_default():
    from urllib.robotparser import RobotFileParser

    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /secret", "Allow: /"])
    assert _robots_allowed(rp, "https://example.com/")
    assert not _robots_allowed(rp, "https://example.com/secret/x")


def test_scrape_rejects_bad_url():
    with pytest.raises(ValueError):
        scrape_website("not-a-url")
    with pytest.raises(ValueError):
        scrape_website("ftp://example.com")


def test_scrape_respects_max_pages(monkeypatch):
    """Mock HTTP so crawl stops at max_pages without network."""
    html_home = """
    <html><head><title>خانه</title></head><body>
    <main>محتوای صفحه اصلی سایت آزمایشی جواد با متن کافی برای پذیرش.
    این پاراگراف تکراری است تا طول متن از چهل کاراکتر بیشتر شود.</main>
    <a href="/p1">یک</a><a href="/p2">دو</a><a href="/p3">سه</a>
    </body></html>
    """
    html_page = """
    <html><head><title>صفحه</title></head><body>
    <main>محتوای صفحه داخلی آزمایشی با متن کافی برای یادگیری در پایگاه دانش.
    ادامه متن برای عبور از آستانه حداقل طول.</main>
    </body></html>
    """

    def fake_fetch(client, url):
        if url.rstrip("/").endswith("example.com") or url.endswith("example.com/"):
            return html_home
        return html_page

    monkeypatch.setattr("ollama_rag.scraper._fetch_html", fake_fetch)
    monkeypatch.setattr("ollama_rag.scraper._load_robots", lambda *a, **k: MagicMock(**{"can_fetch.return_value": True}))
    monkeypatch.setattr("ollama_rag.scraper._discover_sitemap_urls", lambda *a, **k: [])

    # Avoid real HTTP client network — patch Client context
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            raise AssertionError("should not hit network for robots/sitemap in this test")

    monkeypatch.setattr("ollama_rag.scraper.httpx.Client", FakeClient)

    result = scrape_website(
        "https://example.com/",
        max_depth=2,
        max_pages=2,
        use_sitemap=False,
        respect_robots=False,
    )
    assert result["pages"] <= 2
    assert result["chars"] > 0
