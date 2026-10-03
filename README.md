# vayudrishti-arch.github.io
Worldwide provider weather dashboard with India-scoped learned rainfall correction by Anubhav Chakraborty

Live app: https://vayudrishti-arch.github.io/app/

VayuDrishti is a research product, not an official warning service. For official alerts use IMD or your local authorities.

## Data and privacy

Forecast and geocoding requests go directly from the browser to Open-Meteo. Search terms and selected coordinates are sent to that provider. The one exception is the "Live alerts & news" card: the selected place name and country go to a small Cloudflare Worker proxy, which queries SerpApi (Google News) and returns headlines (see the SerpApi section below). The dashboard does not request device location, track visitors or poll forecasts automatically. Forecast response caching is temporary in memory; the theme preference is stored locally.

- [GFS API](https://open-meteo.com/en/docs/gfs-api)
- [Geocoding API](https://open-meteo.com/en/docs/geocoding-api)
- [SerpApi Google News API](https://serpapi.com/google-news-api)
- [Fixed-lead historical forecast archive](https://open-meteo.com/en/docs/previous-runs-api)
- [ERA5 historical weather reference](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo terms](https://open-meteo.com/en/terms)

Weather data attribution: Open-Meteo, CC BY 4.0. Location data: GeoNames via Open-Meteo. GFS rainfall values are modified by the fitted correction where available. This is non-commercial research use; no endorsement is implied. A repository code license does not replace source-data terms.

## Live alerts & news (SerpApi)

After a place is loaded, the dashboard shows a "Live alerts & news" card. It is a Google News search (via [SerpApi](https://serpapi.com/)) for recent weather alert, advisory, warning and rain items about that place. These are news articles, not official warnings; check IMD or local authorities.

- The browser never holds a SerpApi key. It calls a small Cloudflare Worker (`serpapi-proxy/`) that holds the key as a secret, builds the query itself, and returns only title, source, date, link and a short snippet.
- Only the place name and country are sent to the proxy and SerpApi. The card loads after a place is selected, not on page load.
- Results are cached per place per day (about 6 h TTL). The free SerpApi plan allows 250 searches a month, so the Worker stops at 230 and the card then stays hidden. Per-IP limits and CORS limited to the Pages origin apply.
- If the proxy is unreachable, over quota or returns nothing, the card stays hidden and the rest of the app is unaffected.
- Deployed as a Cloudflare Worker (`vayudrishti-serpapi`); the `SERPAPI_KEY` secret and a KV counter binding `COUNTERS` are set in Cloudflare. To redeploy elsewhere: set the secret, bind a KV namespace named `COUNTERS`, and point `NEWS_PROXY` in `app/index.html` at the Worker URL.
