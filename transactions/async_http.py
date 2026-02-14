import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp


class ApiError(Exception):
    """Base exception for API errors."""

    pass


class AuthenticationError(ApiError):
    """Raised when authentication fails."""

    pass


class OrderError(ApiError):
    """Raised when order operations fail."""

    pass


class AsyncHTTPClient:
    def __init__(self, base_url: str, timeout: int = 30, retry_count: int = 1):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # self.retry_count = retry_count # Not used currently
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Lazy-create session on first use."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.5)

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make async HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            headers: Additional headers
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            ApiError: On request failure
        """
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {"Content-Type": "application/json"}

        if headers:
            request_headers.update(headers)

        last_error = None
        print(f"Method: {method.upper()}\nurl: {url}\njson: {data}\nheaders: {request_headers}\nparams: {params}")
        try:
            async with self._session.request(
                method=method.upper(),
                url=url,
                json=data,
                headers=request_headers,
                params=params,
            ) as response:
                response.raise_for_status()

                if response.content_length == 0:
                    return {}

                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    text = await response.text()
                    return {"response": text} if text else {}
        except aiohttp.ClientResponseError as e:
            last_error = e
            raise ApiError(f"Request failed: {e.status} {e.message}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise OrderError(f"Request failed: {e.status} {e.message}")
