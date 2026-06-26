"""Shared utilities: time/UTC, HTTP session, URL canonicalization, hashing, logging."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, parse_qs, urlencode

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 is vendored by requests; Retry path differs across versions
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_log_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _log_configured
    if not _log_configured:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _log_configured = True
    return logging.getLogger("mti")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    """ISO-8601 with a trailing Z, second precision."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_dt(value) -> datetime | None:
    """Parse a feed date (struct_time, RFC822/ISO string) into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, time.struct_time):
        return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    # RFC 822 (RSS) e.g. "Thu, 25 Jun 2026 22:58:00 +0300"
    try:
        dt = parsedate_to_datetime(s)
        if dt is not None:
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    # ISO 8601
    iso = s.replace("Z", "+00:00")
    for candidate in (iso, iso[:19]):
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def age_hours(dt: datetime, now: datetime | None = None) -> float:
    now = now or utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now - dt).total_seconds() / 3600.0)


def bucket_id(dt: datetime, step_hours: int = 2) -> str:
    """Stable run id for an N-hour bucket, e.g. 2026-06-26T04Z."""
    h = (dt.hour // step_hours) * step_hours
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT") + f"{h:02d}Z"


def bucket_time(dt: datetime, step_hours: int = 2) -> datetime:
    h = (dt.hour // step_hours) * step_hours
    return dt.astimezone(timezone.utc).replace(hour=h, minute=0, second=0, microsecond=0)


def session(user_agent: str | None = None, timeout: int = 20) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        "Accept-Language": "en,ar;q=0.8,fa;q=0.6,he;q=0.6,tr;q=0.6",
    })
    if Retry is not None:
        retry = Retry(
            total=3, connect=3, read=3, backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    s.request_timeout = timeout  # convenience attribute
    return s


def http_get(sess: requests.Session, url: str, timeout: int | None = None) -> requests.Response:
    return sess.get(url, timeout=timeout or getattr(sess, "request_timeout", 20), allow_redirects=True)


# ---- URL / title normalization for de-duplication ----

_GNEWS_HOSTS = ("news.google.com",)
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "fbclid", "gclid", "ref", "cmpid", "amp"}


def canon_url(u: str) -> str:
    """Best-effort canonical URL: drop tracking params, lower host.
    Google News links are opaque redirects; we keep them but strip params."""
    if not u:
        return ""
    try:
        p = urlparse(u)
    except Exception:
        return u
    host = (p.netloc or "").lower()
    if any(h in host for h in _GNEWS_HOSTS):
        # GNews redirect URL is unique per article path; keep path only
        return f"{host}{p.path}"
    q = parse_qs(p.query)
    q = {k: v for k, v in q.items() if k.lower() not in _TRACKING_PARAMS}
    query = urlencode(sorted((k, vv) for k, vs in q.items() for vv in vs))
    path = p.path.rstrip("/")
    return f"{host}{path}" + (f"?{query}" if query else "")


_SOURCE_SUFFIX = re.compile(r"\s+[-–—|]\s+[^-–—|]{2,40}$")
_WS = re.compile(r"\s+")


def strip_source_suffix(title: str) -> str:
    """Google News appends ' - Outlet'; strip a trailing ' - Source' clause once."""
    if not title:
        return ""
    return _SOURCE_SUFFIX.sub("", title).strip()


def title_norm(title: str) -> str:
    t = strip_source_suffix(title or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return _WS.sub(" ", t).strip()


def title_key(title: str) -> str:
    return hashlib.sha1(title_norm(title).encode("utf-8")).hexdigest()


def token_set(title: str) -> set[str]:
    return set(title_norm(title).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default
