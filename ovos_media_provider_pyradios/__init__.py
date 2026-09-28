"""OVOS MediaProvider plugin for radio-browser.info (via ``pyradios``).

Replaces the deprecated OCP search skill ``ovos-skill-pyradios``. Instead of
answering ``ovos.common_play.query`` over the bus, this provider is loaded
in-process by the OCP pipeline and its :meth:`search` is called directly. The
provider serves radio stations; any request it cannot satisfy (no query, the
backend unreachable, …) yields an empty list.

``pyradios`` is a thin client over the `radio-browser.info
<https://www.radio-browser.info>`_ HTTP API. It does **not** ship a
``mediavocab`` bridge (no ``mediavocab_bridge``/``converters`` module and it is
not part of mediavocab's consumer contract set), so this plugin constructs
:class:`mediavocab.Release` objects itself from the raw station dicts returned
by the API.

Each radio-browser station is a live-linear broadcast, modelled per mediavocab
axiom 8 as a :class:`mediavocab.Work` with ``MediaType.RADIO`` plus a single
``StreamMode.CONTINUOUS`` :class:`mediavocab.Release` pointing at the resolved
stream URL.

Optional plugin config:
--------------
The plugin is configured via the ``mycroft.conf`` file under the OCP media 
providers pipeline key (``ovos-ocp-pipeline-plugin -> media_providers -> pyradios``).

Supported configuration keys:
- ``max_results`` (int): Maximum number of search results returned from radio-browser (default: 10).
- ``favorites`` (list[dict]): Local station overrides with custom stream URLs.
  Each entry supports:
  - ``name`` (str): Canonical station name.
  - ``aliases`` (list[str]): Alternative spoken triggers (case/punctuation-insensitive).
  - ``url`` (str): Direct stream URL.
  - ``genres`` (list[str], optional): Content genres (e.g., ``["pop"]``).
  - ``image`` (str, optional): Custom logo/favicon URL. Fetched via API if omitted.
  - ``codec`` / ``bitrate`` (str/int, optional): Audio format details.
- ``aliases`` (dict[str, str]): Spoken alias mappings to rewrite search queries 
  before querying the radio-browser API (``"spoken query": "target query"``).

Example configuration (``mycroft.conf``):

.. code-block:: json

    {
      "ovos-ocp-pipeline-plugin": {
        "media_providers": {
          "youtube": {
            "max_results": 10
          },
          "pyradios": {
            "max_results": 10,
            "favorites": [
              {
                "name": "Radio 538",
                "aliases": ["538", "radio five three eight", "my favorite radio station"],
                "url": "http://playerservices.streamtheworld.com/api/livestream-redirect/RADIO538.mp3",
                "genres": ["pop"]
              }
            ],
            "aliases": {
              "sky radio": "Sky Radio 101 FM"
            }
          }
        }
      }
    }
"""

import re
from typing import ClassVar, List, Optional, Set

from ovos_utils.log import LOG

from mediavocab import MediaType, Release, Signals, Work
from ovos_plugin_manager.templates.media_provider import MediaProvider

from ovos_media_provider_pyradios.version import __version__  # noqa: F401


def _norm(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def station_to_release(station: dict) -> Optional[Release]:
    """Build a :class:`mediavocab.Release` from a radio-browser station dict.

    radio-browser stations expose (among others) the keys ``name``,
    ``url_resolved``/``url``, ``favicon``, ``tags`` (comma-separated string),
    ``codec``, ``bitrate``, ``language``, ``countrycode``, ``stationuuid`` and
    a ``0.0``–``1.0``-ish vote/click signal. Returns ``None`` for a station
    with neither a name nor a playable URL.
    """
    name = (station.get("name") or "").strip()
    uri = (station.get("url_resolved") or station.get("url") or "").strip()
    if not name or not uri:
        return None

    raw_tags = station.get("tags") or ""
    if isinstance(raw_tags, (list, tuple)):
        tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
    else:
        tags = [t.strip().lower() for t in str(raw_tags).split(",") if t.strip()]

    # pydantic models reject None for these fields: use empty strings instead
    language = (station.get("language") or "").strip()
    country = (station.get("countrycode") or "").strip()

    external_ids: dict = {}
    if station.get("stationuuid"):
        external_ids["radio_browser_uuid"] = str(station["stationuuid"])

    work = Work(
        title=name,
        media_type=MediaType.RADIO,
        language=language,
        broadcaster_country=country,
        content_genres=tags,
        external_ids=dict(external_ids),
    )

    bitrate = None
    raw_bitrate = station.get("bitrate")
    if raw_bitrate:
        try:
            # mediavocab's Release.bitrate is a free-form string (e.g. "128")
            bitrate = str(int(raw_bitrate)) if int(raw_bitrate) else None
        except (TypeError, ValueError):
            bitrate = str(raw_bitrate).strip() or None

    # only pass optional fields when set, so pydantic never sees None
    release_kwargs: dict = {}
    codec = (station.get("codec") or "").strip()
    if codec:
        release_kwargs["codec"] = codec
    if bitrate:
        release_kwargs["bitrate"] = bitrate

    return Release(
        work=work,
        uri=uri,
        image=(station.get("favicon") or "").strip(),
        platform="radio-browser",
        match_confidence=_station_confidence(station),
        external_ids=dict(external_ids),
        **release_kwargs,
    )


def _station_confidence(station: dict) -> float:
    """Derive a coarse ``0.0``–``1.0`` ranking signal from a station's
    radio-browser popularity. Uses ``clickcount`` (saturating) so more popular
    stations float to the top across a multi-provider search."""
    try:
        clicks = int(station.get("clickcount") or 0)
    except (TypeError, ValueError):
        clicks = 0
    # saturate: 0 clicks -> 0.5 baseline, ~1000+ clicks -> ~1.0
    return round(min(1.0, 0.5 + clicks / 2000.0), 3)


class PyRadiosMediaProvider(MediaProvider):
    """Search radio-browser.info and return ``mediavocab.Release`` playables.

    Serves live-linear radio stations (``MediaType.RADIO``, consumed as audio).
    """

    name: ClassVar[str] = "pyradios"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        # max results per search, overridable via plugin config
        self.max_results: int = int(self.config.get("max_results", 10))
        # user-defined stations with their own stream URL and spoken aliases
        self.favorites: list = list(self.config.get("favorites") or [])
        # spoken alias -> name to search for in radio-browser
        self.aliases: dict = {_norm(k): v for k, v in
                              (self.config.get("aliases") or {}).items()}
        self._rb = None

    @property
    def rb(self):
        """Lazily-constructed ``pyradios.RadioBrowser`` client (it pings the
        API's DNS pool at construction time, so build it on first use)."""
        if self._rb is None:
            from pyradios import RadioBrowser
            self._rb = RadioBrowser()
        return self._rb

    def _get_favorite_image(self, fav: dict) -> str:
        """Fetch the image from the favorite config, or lookup the favicon via radio-browser."""
        image = (fav.get("image") or "").strip()
        if image:
            return image

        # If no image URL is provided in the config, look it up via the API
        try:
            results = self.rb.search(name=fav["name"], limit=1)
            if results and results[0].get("favicon"):
                return results[0]["favicon"].strip()
        except Exception:
            LOG.warning(f"Failed to fetch alternative logo for favorite: {fav.get('name')!r}")

        return ""

    def _match_favorites(self, query: str) -> List[Release]:
        """Return Releases for configured favorites whose name or alias
        matches ``query`` (case- and punctuation-insensitive)."""
        q = _norm(query)
        if not q:
            return []
        out: List[Release] = []
        for fav in self.favorites:
            if not fav.get("name") or not fav.get("url"):
                continue
            names = [fav["name"]] + list(fav.get("aliases") or [])
            if q not in {_norm(n) for n in names}:
                continue

            # Fetch image from favorite or retrieve it via the RadioBrowser API
            image = self._get_favorite_image(fav)

            station = {
                "name": fav["name"],
                "url": fav["url"],
                "favicon": image,
                "tags": fav.get("genres", []),
                "codec": fav.get("codec"),
                "bitrate": fav.get("bitrate"),
                "clickcount": 10 ** 6,  # saturates confidence at 1.0
            }
            try:
                rel = station_to_release(station)
            except Exception:
                LOG.exception(f"Invalid favorite in config: {fav.get('name')!r}")
                continue
            if rel is not None:
                out.append(rel)
        return out

    def search(self, signals: Signals, lang: str = "en-us", *,
               supported_playback_types: Optional[Set[str]] = None,
               blocked_genres: Optional[Set[str]] = None,
               region: Optional[str] = None,
               session_id: Optional[str] = None) -> List[Release]:
        """Search radio-browser for ``signals.title`` and return Releases.

        A configured favorite whose name/alias matches the title is returned
        directly, without querying the API. Otherwise the title is looked up
        in the ``aliases`` config (alias -> search name) and then searched by
        station name; if ``signals`` carry genre/content tags those are passed
        through as a radio-browser ``tag_list`` filter. Each station dict is
        mapped to a :class:`mediavocab.Release` via
        :func:`station_to_release`. Returns ``[]`` when the request carries
        neither a title nor genres, or when the radio-browser API is
        unreachable.
        """
        query = (signals.title or "").strip()
        tags = [str(g).strip() for g in (signals.content_genres or []) if str(g).strip()]
        if not query and not tags:
            return []

        favs = self._match_favorites(query)
        if favs:
            return favs
        query = self.aliases.get(_norm(query), query)

        kwargs: dict = {
            "limit": self.max_results,
            "hidebroken": True,
            "order": "clickcount",
            "reverse": True,
        }
        if query:
            kwargs["name"] = query
        if tags:
            kwargs["tag_list"] = ",".join(tags)

        releases: List[Release] = []
        try:
            stations = self.rb.search(**kwargs) or []
        except Exception:
            LOG.exception(f"radio-browser search failed for query: {query!r}")
            return []

        for station in stations:
            try:
                rel = station_to_release(station)
                if rel is not None:
                    releases.append(rel)
            except Exception:
                LOG.exception("Failed to convert radio-browser station to Release")
        return releases
