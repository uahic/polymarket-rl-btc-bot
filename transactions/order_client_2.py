"""
Minimalistic Polymarket Order Client

A simplified async client for Polymarket's CLOB API based on official documentation.
Focuses on core order operations without complex abstractions.

Documentation: https://docs.polymarket.com/developers/CLOB/orders/create-order

Usage:
    from transactions.order_client_2 import PolymarketClient

    client = PolymarketClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        passphrase="your_passphrase",
        address="0x..."
    )

    # Place an order
    response = await client.place_order(
        signed_order=signed_order_dict,
        order_type="GTC"
    )

    # Cancel an order
    await client.cancel_order(order_id)

    # Get order book
    book = await client.get_order_book(token_id)
"""

import time
import hmac
import hashlib
import base64
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import aiohttp


@dataclass
class Credentials:
    """API credentials for L2 authentication."""
    api_key: str
    api_secret: str
    passphrase: str
    address: str

    def is_valid(self) -> bool:
        """Check if all credentials are provided."""
        return bool(self.api_key and self.api_secret and self.passphrase and self.address)


class PolymarketClient:
    """
    Minimalistic async client for Polymarket CLOB API.

    Implements core functionality:
    - Order placement (POST /order)
    - Order cancellation (DELETE /order)
    - Order book queries (GET /book)
    - Market prices (GET /price)
    - L2 HMAC authentication
    """

    CLOB_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        address: str,
        timeout: int = 30
    ):
        """
        Initialize client with API credentials.

        Args:
            api_key: L2 API key
            api_secret: L2 API secret (base64 encoded)
            passphrase: L2 passphrase
            address: Ethereum address (owner)
            timeout: Request timeout in seconds
        """
        self.creds = Credentials(api_key, api_secret, passphrase, address)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Generate L2 authentication headers using HMAC-SHA256.

        Per Polymarket documentation:
        - Message format: timestamp + method + path + body
        - Signature: HMAC-SHA256 with base64-decoded secret
        - Result: base64-encoded signature

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Request path (e.g., "/order")
            body: JSON request body (empty string if no body)

        Returns:
            Dictionary of authentication headers
        """
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}{body}"

        # Decode base64 secret and sign
        secret = base64.b64decode(self.creds.api_secret)
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()

        return {
            "POLY_ADDRESS": self.creds.address,
            "POLY_API_KEY": self.creds.api_key,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": self.creds.passphrase,
            "POLY_SIGNATURE": signature_b64,
        }

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Any] = None,
        params: Optional[Dict] = None,
        authenticated: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request to CLOB API.

        Args:
            method: HTTP method
            path: API path
            data: Request body (will be JSON encoded)
            params: Query parameters
            authenticated: Whether to add auth headers

        Returns:
            Response JSON

        Raises:
            aiohttp.ClientError: On request failure
        """
        session = await self._get_session()
        url = f"{self.CLOB_URL}{path}"

        headers = {"Content-Type": "application/json"}

        # Add authentication headers if required
        if authenticated:
            body_json = json.dumps(data, separators=(',', ':')) if data else ""
            headers.update(self._sign_request(method, path, body_json))

        async with session.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers
        ) as response:
            response.raise_for_status()

            # Handle empty responses
            if response.content_length == 0:
                return {}

            return await response.json()

    # ============================================================================
    # Order Operations
    # ============================================================================

    async def place_order(
        self,
        signed_order: Dict[str, Any],
        order_type: str = "GTC",
        post_only: bool = False
    ) -> Dict[str, Any]:
        """
        Place a signed order on the CLOB.

        Per documentation (POST /order):
        - Requires signed order with EIP-712 signature
        - Order types: GTC (Good-Til-Cancelled), GTD (Good-Til-Date), FOK (Fill-Or-Kill)
        - postOnly prevents immediate matching (order rejected if marketable)

        Args:
            signed_order: Signed order object with signature
            order_type: "GTC", "GTD", or "FOK"
            post_only: Prevent immediate matching (default False)

        Returns:
            Response with:
            - success: bool
            - errorMsg: str (empty if successful)
            - orderId: str (if successful)
            - orderHashes: List[str] (settlement tx hashes if matched)

        Raises:
            aiohttp.ClientError: On API error
        """
        body = {
            "order": signed_order,
            "owner": self.creds.address,
            "orderType": order_type,
        }

        if post_only:
            body["postOnly"] = True

        return await self._request("POST", "/order", data=body)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel a single order by ID.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation response
        """
        body = {"orderID": order_id}
        return await self._request("DELETE", "/order", data=body)

    async def cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """
        Cancel multiple orders.

        Args:
            order_ids: List of order IDs

        Returns:
            Response with canceled and not_canceled lists
        """
        return await self._request("DELETE", "/orders", data=order_ids)

    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders for the account.

        Returns:
            Response with canceled and not_canceled lists
        """
        return await self._request("DELETE", "/cancel-all")

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel orders for a specific market or asset.

        Args:
            market: Market condition ID
            asset_id: Token/asset ID

        Returns:
            Response with canceled and not_canceled lists
        """
        body = {}
        if market:
            body["market"] = market
        if asset_id:
            body["asset_id"] = asset_id

        return await self._request("DELETE", "/cancel-market-orders", data=body if body else None)

    # ============================================================================
    # Market Data (Public)
    # ============================================================================

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book with bids and asks
        """
        return await self._request(
            "GET",
            "/book",
            params={"token_id": token_id},
            authenticated=False
        )

    async def get_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data (mid, bid, ask)
        """
        return await self._request(
            "GET",
            "/price",
            params={"token_id": token_id},
            authenticated=False
        )

    # ============================================================================
    # Account Data (Authenticated)
    # ============================================================================

    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the account.

        Returns:
            List of orders
        """
        result = await self._request("GET", "/data/orders")

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get specific order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details
        """
        return await self._request("GET", f"/data/order/{order_id}")

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history for the account.

        Args:
            token_id: Filter by token (optional)
            limit: Max number of trades

        Returns:
            List of trades
        """
        params: Dict[str, Any] = {"limit": limit}
        if token_id:
            params["token_id"] = token_id

        result = await self._request("GET", "/data/trades", params=params)

        # Handle paginated response
        if isinstance(result, dict) and "data" in result:
            return result.get("data", [])
        return result if isinstance(result, list) else []


# ============================================================================
# Example Usage
# ============================================================================

async def example():
    """Example usage of the minimalistic client."""
    client = PolymarketClient(
        api_key="your_api_key",
        api_secret="your_api_secret",
        passphrase="your_passphrase",
        address="0x..."
    )

    try:
        # Get order book (public, no auth)
        book = await client.get_order_book(token_id="12345")
        print(f"Order book: {book}")

        # Get current price (public, no auth)
        price = await client.get_price(token_id="12345")
        print(f"Price: {price}")

        # Place an order (requires signed order)
        # Note: Order must be signed with EIP-712 signature first
        signed_order = {
            "salt": "12345...",
            "maker": "0x...",
            "signer": "0x...",
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": "12345",
            "makerAmount": "1000000",  # USDC amount (6 decimals)
            "takerAmount": "1000000",  # Token amount
            "expiration": "0",  # 0 = no expiration
            "nonce": "0",
            "feeRateBps": "0",
            "side": "BUY",
            "signatureType": "2",
            "signature": "0x..."
        }

        response = await client.place_order(signed_order, order_type="GTC")

        if response.get("success"):
            print(f"Order placed: {response.get('orderId')}")
        else:
            print(f"Order failed: {response.get('errorMsg')}")

        # Get open orders
        orders = await client.get_orders()
        print(f"Open orders: {len(orders)}")

        # Cancel an order
        if orders:
            order_id = orders[0].get("id")
            await client.cancel_order(order_id)
            print(f"Canceled order: {order_id}")

    finally:
        await client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example())
