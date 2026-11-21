import re
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_HASHED_ASSET_RE = re.compile(
    r"^/(?:[^/]+\.[0-9a-f]{8,}|[0-9a-f]{16,})\.(?:js|css|png|ico|woff2?|svg)$"
)

_CONDITIONAL_REQ_HEADERS = {b"if-none-match", b"if-modified-since"}
_STRIP_NO_CACHE_RESP_HEADERS = {
    b"cache-control",
    b"pragma",
    b"expires",
    b"etag",
    b"last-modified",
}
_STRIP_IMMUTABLE_RESP_HEADERS = {
    b"cache-control",
    b"pragma",
    b"expires",
}

_NO_CACHE_HEADERS = [
    (b"cache-control", b"no-store, no-cache, must-revalidate, max-age=0"),
    (b"pragma", b"no-cache"),
    (b"expires", b"0"),
]
_IMMUTABLE_HEADERS = [
    (b"cache-control", b"public, max-age=31536000, immutable"),
]


class CacheControlMiddleware:
    """
    Outer ASGI middleware enforcing strict no-store/no-cache headers on HTML, API JSON,
    user uploads, and unhashed assets while allowing immutable caching on content-hashed
    top-level Webpack bundles.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        is_immutable_candidate = bool(_HASHED_ASSET_RE.match(path))

        forwarded_scope = scope
        if not is_immutable_candidate:
            raw_headers = scope.get("headers") or []
            filtered_headers = [
                (k, v) for k, v in raw_headers if k.lower() not in _CONDITIONAL_REQ_HEADERS
            ]
            if len(filtered_headers) != len(raw_headers):
                forwarded_scope = dict(scope)
                forwarded_scope["headers"] = filtered_headers

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status = int(message.get("status", 200))
                headers = list(message.get("headers") or [])
                if is_immutable_candidate and status < 400:
                    cleaned = [
                        (k, v)
                        for k, v in headers
                        if k.lower() not in _STRIP_IMMUTABLE_RESP_HEADERS
                    ]
                    cleaned.extend(_IMMUTABLE_HEADERS)
                else:
                    cleaned = [
                        (k, v)
                        for k, v in headers
                        if k.lower() not in _STRIP_NO_CACHE_RESP_HEADERS
                    ]
                    cleaned.extend(_NO_CACHE_HEADERS)
                message = {**message, "headers": cleaned}
            await send(message)

        await self.app(forwarded_scope, receive, send_wrapper)
