"""
Async Order Client Module - Async API Clients for Polymarket

Provides async clients for interacting with:
- CLOB (Central Limit Order Book) API
- Builder Relayer API

Features:
- Non-blocking async HTTP calls using aiohttp
- Gasless transactions via Builder Program
- HMAC authentication for Builder APIs
- Automatic retry and error handling
- Session reuse for performance

Example:
    from transactions.order_client import AsyncClobClient, AsyncRelayerClient

    async def main():
        clob = AsyncClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            signature_type=2,
            funder="0x..."
        )

        try:
            # Use the client
            orders = await clob.get_open_orders()

        finally:
            # Always close the session
            await clob.close()

    asyncio.run(main())
"""

import asyncio
import time
import hmac
import hashlib
import base64
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp

from config import BuilderConfig


class ApiError(Exception):
    """Base exception for API errors."""
    pass


class AuthenticationError(ApiError):
    """Raised when authentication fails."""
    pass


class OrderError(ApiError):
    """Raised when order operations fail."""
    pass


@dataclass
class ApiCredentials:
    """User-level API credentials for CLOB."""
    api_key: str
    secret: str
    passphrase: str

    @classmethod
    def load(cls, filepath: str) -> "ApiCredentials":
        """Load credentials from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            api_key=data.get("apiKey", ""),
            secret=data.get("secret", ""),
            passphrase=data.get("passphrase", ""),
        )

    def is_valid(self) -> bool:
        """Check if credentials are valid."""
        return bool(self.api_key and self.secret and self.passphrase)


class AsyncApiClient:
    """
    Base async HTTP client with common functionality.

    Provides:
    - Automatic JSON handling
    - Request/response logging
    - Error handling
    - Session management with reuse
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retry_count: int = 3
    ):
        """
        Initialize async API client.

        Args:
            base_url: Base URL for all requests
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_count = retry_count
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Lazy-create session on first use."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the session gracefully."""
        if self._session and not self._session.closed:
            await self._session.close()
            # Give time for connections to close
            await asyncio.sleep(0.250)

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Any] = None,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None
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
        for attempt in range(self.retry_count):
            try:
                async with self._session.request(
                    method=method.upper(),
                    url=url,
                    json=data,
                    headers=request_headers,
                    params=params
                ) as response:
                    response.raise_for_status()

                    # Handle empty responses
                    if response.content_length == 0:
                        return {}

                    # Parse JSON response
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        # Response is not JSON
                        text = await response.text()
                        return {"response": text} if text else {}

            except aiohttp.ClientResponseError as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Last attempt - raise with details
                    raise ApiError(f"Request failed: {e.status} {e.message}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

        raise ApiError(f"Request failed after {self.retry_count} attempts: {last_error}")


class AsyncClobClient(AsyncApiClient):
    """
    Async client for Polymarket CLOB (Central Limit Order Book) API.

    Features:
    - Order placement and cancellation
    - Order book queries
    - Trade history
    - Builder attribution support
    - Non-blocking async operations

    Example:
        client = AsyncClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            signature_type=2,
            funder="0x..."
        )

        try:
            orders = await client.get_open_orders()
        finally:
            await client.close()
    """

    def __init__(
        self,
        host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        signature_type: int = 2,
        funder: str = "",
        api_creds: Optional[ApiCredentials] = None,
        builder_creds: Optional[BuilderConfig] = None,
        timeout: int = 30
    ):
        """
        Initialize async CLOB client.

        Args:
            host: CLOB API host
            chain_id: Chain ID (137 for Polygon mainnet)
            signature_type: Signature type (2 = Gnosis Safe)
            funder: Funder/Safe address
            api_creds: User API credentials (optional)
            builder_creds: Builder credentials for attribution (optional)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.host = host
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.api_creds = api_creds
        self.builder_creds = builder_creds

    def _build_hmac_signature(
        self,
        secret: bytes,
        message: str,
        output_format: str = "base64"
    ) -> str:
        """
        Build HMAC-SHA256 signature.

        Args:
            secret: Secret key as bytes
            message: Message to sign
            output_format: "base64" or "hex"

        Returns:
            Signature as string
        """
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        if output_format == "base64":
            return base64.b64encode(signature.digest()).decode()
        return signature.hexdigest()

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Build authentication headers.

        Supports both user API credentials and Builder credentials.

        Args:
            method: HTTP method
            path: Request path
            body: Request body (JSON string)

        Returns:
            Dictionary of headers
        """
        headers = {}
        timestamp = str(int(time.time()))

        # Builder HMAC authentication (uses hex encoding)
        if self.builder_creds and self.builder_creds.is_configured():
            message = f"{timestamp}{method}{path}{body}"
            signature = self._build_hmac_signature(
                self.builder_creds.api_secret.encode(),
                message,
                output_format="hex"
            )

            headers.update({
                "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
                "POLY_BUILDER_TIMESTAMP": timestamp,
                "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
                "POLY_BUILDER_SIGNATURE": signature,
            })

        # User API credentials (L2 authentication - uses base64 encoding)
        if self.api_creds and self.api_creds.is_valid():
            message = f"{timestamp}{method}{path}{body}"

            # Secret is base64-encoded, must decode before signing
            secret = base64.b64decode(self.api_creds.secret)
            signature = self._build_hmac_signature(
                secret,
                message,
                output_format="base64"
            )

            headers.update({
                "POLY_ADDRESS": self.funder,
                "POLY_API_KEY": self.api_creds.api_key,
                "POLY_TIMESTAMP": timestamp,
                "POLY_PASSPHRASE": self.api_creds.passphrase,
                "POLY_SIGNATURE": signature,
            })

        return headers

    async def derive_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Derive L2 API credentials using L1 EIP-712 authentication.

        This is required to access authenticated endpoints like
        /orders and /trades.

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        timestamp = str(int(time.time()))

        # Sign the auth message using EIP-712
        auth_signature = signer.sign_auth_message(timestamp=timestamp, nonce=nonce)

        # L1 headers
        headers = {
            "POLY_ADDRESS": signer.address,
            "POLY_SIGNATURE": auth_signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

        response = await self._request("GET", "/auth/derive-api-key", headers=headers)

        return ApiCredentials(
            api_key=response.get("apiKey", ""),
            secret=response.get("secret", ""),
            passphrase=response.get("passphrase", ""),
        )

    async def create_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Create new L2 API credentials using L1 EIP-712 authentication.

        Use this if derive_api_key fails (first time setup).

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        timestamp = str(int(time.time()))

        # Sign the auth message using EIP-712
        auth_signature = signer.sign_auth_message(timestamp=timestamp, nonce=nonce)

        # L1 headers
        headers = {
            "POLY_ADDRESS": signer.address,
            "POLY_SIGNATURE": auth_signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

        response = await self._request("POST", "/auth/api-key", headers=headers)

        return ApiCredentials(
            api_key=response.get("apiKey", ""),
            secret=response.get("secret", ""),
            passphrase=response.get("passphrase", ""),
        )

    async def create_or_derive_api_key(self, signer: "OrderSigner", nonce: int = 0) -> ApiCredentials:
        """
        Create API credentials if not exists, otherwise derive them.

        Args:
            signer: OrderSigner instance with private key
            nonce: Nonce for the auth message (default 0)

        Returns:
            ApiCredentials with api_key, secret, and passphrase
        """
        try:
            return await self.create_api_key(signer, nonce)
        except Exception:
            return await self.derive_api_key(signer, nonce)

    def set_api_creds(self, creds: ApiCredentials) -> None:
        """Set API credentials for authenticated requests."""
        self.api_creds = creds

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        return await self._request(
            "GET",
            "/book",
            params={"token_id": token_id}
        )

    async def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        return await self._request(
            "GET",
            "/price",
            params={"token_id": token_id}
        )

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders for the funder.

        Returns:
            List of open orders
        """
        endpoint = "/data/orders"
        headers = self._build_headers("GET", endpoint)

        result = await self._request("GET", endpoint, headers=headers)

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details
        """
        endpoint = f"/data/order/{order_id}"
        headers = self._build_headers("GET", endpoint)
        return await self._request("GET", endpoint, headers=headers)

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Filter by token (optional)
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        endpoint = "/data/trades"
        headers = self._build_headers("GET", endpoint)
        params: Dict[str, Any] = {"limit": limit}
        if token_id:
            params["token_id"] = token_id

        result = await self._request("GET", endpoint, headers=headers, params=params)

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    async def post_order(
        self,
        signed_order: Dict[str, Any],
        order_type: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Submit a signed order asynchronously.

        Args:
            signed_order: Order with signature
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            Response with order ID and status
        """
        endpoint = "/order"

        # Build request body
        # Note: signature must be inside the order object per API spec
        order_obj = signed_order.get("order", signed_order)
        if "signature" in signed_order:
            order_obj["signature"] = signed_order["signature"]

        body = {
            "order": order_obj,
            "owner": self.funder,
            "orderType": order_type,
        }

        # POST /order uses EIP-712 signature within the order itself for authentication
        # Builder headers are ONLY for attribution, not authentication
        # Don't send builder headers for now - they may be causing 401 errors
        headers = None

        return await self._request("POST", endpoint, data=body, headers=headers)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        endpoint = "/order"
        body = {"orderID": order_id}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request("DELETE", endpoint, data=body, headers=headers)

    async def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """
        Cancel multiple orders by their IDs.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/orders"
        body_json = json.dumps(order_ids, separators=(',', ':'))
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request("DELETE", endpoint, data=order_ids, headers=headers)

    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-all"
        headers = self._build_headers("DELETE", endpoint)

        return await self._request("DELETE", endpoint, headers=headers)

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-market-orders"
        body = {}

        if market:
            body["market"] = market
        if asset_id:
            body["asset_id"] = asset_id

        body_json = json.dumps(body, separators=(',', ':')) if body else ""
        headers = self._build_headers("DELETE", endpoint, body_json)

        return await self._request("DELETE", endpoint, data=body if body else None, headers=headers)


class AsyncRelayerClient(AsyncApiClient):
    """
    Async client for Builder Relayer API.

    Provides gasless transactions through Polymarket's
    relayer infrastructure.

    Example:
        client = AsyncRelayerClient(
            host="https://relayer-v2.polymarket.com",
            chain_id=137,
            builder_creds=builder_creds
        )

        try:
            result = await client.approve_usdc(safe_address, spender, amount)
        finally:
            await client.close()
    """

    def __init__(
        self,
        host: str = "https://relayer-v2.polymarket.com",
        chain_id: int = 137,
        builder_creds: Optional[BuilderConfig] = None,
        tx_type: str = "SAFE",
        timeout: int = 60
    ):
        """
        Initialize async Relayer client.

        Args:
            host: Relayer API host
            chain_id: Chain ID (137 for Polygon)
            builder_creds: Builder credentials
            tx_type: Transaction type (SAFE or PROXY)
            timeout: Request timeout
        """
        super().__init__(base_url=host, timeout=timeout)
        self.chain_id = chain_id
        self.builder_creds = builder_creds
        self.tx_type = tx_type

    def _build_headers(
        self,
        method: str,
        path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """Build Builder HMAC authentication headers."""
        if not self.builder_creds or not self.builder_creds.is_configured():
            raise AuthenticationError("Builder credentials required for relayer")

        timestamp = str(int(time.time()))

        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.builder_creds.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "POLY_BUILDER_API_KEY": self.builder_creds.api_key,
            "POLY_BUILDER_TIMESTAMP": timestamp,
            "POLY_BUILDER_PASSPHRASE": self.builder_creds.api_passphrase,
            "POLY_BUILDER_SIGNATURE": signature,
        }

    async def deploy_safe(self, safe_address: str) -> Dict[str, Any]:
        """
        Deploy a Safe proxy wallet.

        Args:
            safe_address: The Safe address to deploy

        Returns:
            Deployment transaction response
        """
        endpoint = "/deploy"
        body = {"safeAddress": safe_address}
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request("POST", endpoint, data=body, headers=headers)

    async def approve_usdc(
        self,
        safe_address: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve USDC spending.

        Args:
            safe_address: Safe address
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-usdc"
        body = {
            "safeAddress": safe_address,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request("POST", endpoint, data=body, headers=headers)

    async def approve_token(
        self,
        safe_address: str,
        token_id: str,
        spender: str,
        amount: int
    ) -> Dict[str, Any]:
        """
        Approve an ERC-1155 token.

        Args:
            safe_address: Safe address
            token_id: Token ID
            spender: Spender address
            amount: Amount to approve

        Returns:
            Approval transaction response
        """
        endpoint = "/approve-token"
        body = {
            "safeAddress": safe_address,
            "tokenId": token_id,
            "spender": spender,
            "amount": str(amount),
        }
        body_json = json.dumps(body, separators=(',', ':'))
        headers = self._build_headers("POST", endpoint, body_json)

        return await self._request("POST", endpoint, data=body, headers=headers)
