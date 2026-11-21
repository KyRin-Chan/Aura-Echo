export function withCacheBuster(url: string, token?: string | number): string {
  if (!url || url.startsWith('data:') || url.startsWith('blob:')) {
    return url;
  }
  const value = token !== undefined ? String(token) : String(Date.now());
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}v=${encodeURIComponent(value)}`;
}

export function fetchNoCache(url: string, init?: RequestInit): Promise<Response> {
  return fetch(withCacheBuster(url, Date.now()), {
    ...init,
    cache: 'no-store',
  });
}
