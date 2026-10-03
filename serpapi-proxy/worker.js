// VayuDrishti SerpApi proxy (Cloudflare Worker). Key lives in the SERPAPI_KEY secret only.
const ORIGINS = ['https://vayudrishti-arch.github.io'];
const TTL = 6 * 3600, MONTH_CAP = 230, IP_PER_HOUR = 20;
const cors = o => ({ 'Access-Control-Allow-Origin': o, 'Vary': 'Origin', 'Access-Control-Allow-Methods': 'GET', 'Content-Type': 'application/json; charset=utf-8' });
const clean = (s, n) => String(s || '').replace(/[^\p{L}\p{N} .,'-]/gu, '').trim().slice(0, n);
export default {
  async fetch(req, env) {
    const origin = req.headers.get('Origin') || '';
    const okOrigin = ORIGINS.includes(origin) || (env.ALLOW_LOCAL === '1' && /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin));
    const h = cors(okOrigin ? origin : ORIGINS[0]);
    const out = (o, st = 200) => new Response(JSON.stringify(o), { status: st, headers: h });
    const url = new URL(req.url);
    if (req.method === 'OPTIONS') return new Response(null, { headers: h });
    if (url.pathname !== '/alerts' || req.method !== 'GET') return out({ status: 'error', error: 'not found' }, 404);
    if (!okOrigin) return out({ status: 'error', error: 'origin not allowed' }, 403);
    const place = clean(url.searchParams.get('place'), 60), country = clean(url.searchParams.get('country'), 40);
    if (place.length < 2) return out({ status: 'error', error: 'place required' }, 400);
    const key = 'https://cache.local/' + encodeURIComponent((place + '|' + country).toLowerCase()) + '|' + new Date().toISOString().slice(0, 10);
    const cache = caches.default, hit = await cache.match(key);
    if (hit) return new Response(hit.body, { headers: { ...h, 'X-Cache': 'HIT' } });
    const kv = env.COUNTERS;
    const ip = req.headers.get('CF-Connecting-IP') || 'x', hr = new Date().toISOString().slice(0, 13);
    if (kv) {
      const ik = 'ip:' + ip + ':' + hr, n = +(await kv.get(ik) || 0);
      if (n >= IP_PER_HOUR) return out({ status: 'limited' }, 429);
      await kv.put(ik, String(n + 1), { expirationTtl: 3700 });
      const mk = 'm:' + new Date().toISOString().slice(0, 7), m = +(await kv.get(mk) || 0);
      if (m >= MONTH_CAP) return out({ status: 'quota' });
      await kv.put(mk, String(m + 1), { expirationTtl: 3600 * 24 * 40 });
    }
    const q = `${place}${country ? ' ' + country : ''} weather alert OR advisory OR warning OR rain OR flood when:14d`;
    const u = new URL('https://serpapi.com/search.json');
    u.search = new URLSearchParams({ engine: 'google_news', q, gl: 'in', hl: 'en', api_key: env.SERPAPI_KEY }).toString();
    let j;
    try { const r = await fetch(u); if (!r.ok) return out({ status: 'error', error: 'upstream ' + r.status }, 502); j = await r.json(); }
    catch (e) { return out({ status: 'error', error: 'upstream failed' }, 502); }
    const items = (j.news_results || []).slice(0, 8).map(n => ({
      title: String(n.title || '').slice(0, 200), source: String((n.source && n.source.name) || '').slice(0, 80),
      date: String(n.date || '').slice(0, 40), link: /^https?:\/\//.test(n.link) ? n.link : '', snippet: String(n.snippet || '').slice(0, 240)
    })).filter(n => n.title && n.link);
    const body = JSON.stringify({ status: 'ok', place, items, fetched: new Date().toISOString() });
    const res = new Response(body, { headers: { ...h, 'Cache-Control': 'public, max-age=' + TTL } });
    await cache.put(key, res.clone());
    return res;
  }
};
