import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import time
import hmac
import hashlib
import base64
import json
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .async_http import AsyncHTTPClient
from config import BuilderConfig
from structures.credentials import ClobCredentials
from security.order_signer import OrderSigner
from py_clob_client import ClobClient
from py_clob_client.headers.headers import create_level_1_headers, create_level_2_headers


def create_hmac_signature(secret: bytes, message: str, output_format: str = "base64"):
    """
    Build HMAC-SHA256 signature.

    Args:
        secret: Secret key as bytes
        message: Message to sign
        output_format: "base64", "base64url", or "hex"

    Returns:
        Signature as string
    """
    signature = hmac.new(secret, message.encode(), hashlib.sha256)
    if output_format == "base64":
        return base64.b64encode(signature.digest()).decode()
    elif output_format == "base64url":
        # URL-safe base64 encoding (- and _ instead of + and /)
        return base64.urlsafe_b64encode(signature.digest()).decode()
    return signature.hexdigest()


def create_headers(
    method: str,
    path: str,
    body: str = "",
    funder: str = "",
    builder_creds: Optional[BuilderConfig] = None,
    api_creds: Optional[ClobCredentials] = None,
):
    headers = {}
    timestamp = str(int(time.time()))

    # Create message once for consistency
    message = f"{timestamp}{method}{path}{body}"

    # Builder HMAC authentication (uses hex encoding)
    if builder_creds and builder_creds.is_configured():
        signature = create_hmac_signature(
            builder_creds.api_secret.encode(), message, output_format="hex"
        )
        headers.update(
            {
                "POLY_BUILDER_API_KEY": builder_creds.api_key,
                "POLY_BUILDER_TIMESTAMP": timestamp,
                "POLY_BUILDER_PASSPHRASE": builder_creds.api_passphrase,
                "POLY_BUILDER_SIGNATURE": signature,
            }
        )

    # User API credentials (L2 authentication - uses base64 encoding)
    if api_creds and api_creds.is_valid():
        # The secret from Polymarket's API is base64-encoded
        # According to their API docs, the secret is returned as base64
        secret_str = api_creds.secret

        # The secret from Polymarket contains - and _ so it's URL-safe base64
        # Try URL-safe base64 first since that's what Polymarket returns
        try:
            secret = base64.urlsafe_b64decode(api_creds.secret)
            print(f"[DEBUG] Decoded secret as URL-safe base64, length: {len(secret)} bytes")
        except Exception as e:
            print(f"[DEBUG] URL-safe base64 failed: {e}, trying standard base64")
            try:
                secret = base64.b64decode(api_creds.secret)
                print(f"[DEBUG] Decoded secret as standard base64")
            except Exception as e2:
                print(f"[DEBUG] Standard base64 failed: {e2}, trying hex")
                try:
                    secret = bytes.fromhex(secret_str)
                    print(f"[DEBUG] Decoded as hex")
                except ValueError:
                    print(f"[DEBUG] Hex failed, using as raw string")
                    secret = api_creds.secret.encode()

        # Use URL-safe base64 for the signature (per Polymarket spec)
        signature = create_hmac_signature(secret, message, output_format="base64url")

        # Debug: Show what we're signing
        print(f"[DEBUG] Signing message: '{message[:100]}...' (len={len(message)})")
        print(f"[DEBUG] Generated signature: '{signature}'")

        headers.update(
            {
                "POLY_ADDRESS": funder,  # Use same address format as credential derivation
                "POLY_API_KEY": api_creds.api_key,
                "POLY_TIMESTAMP": timestamp,
                "POLY_PASSPHRASE": api_creds.passphrase,
                "POLY_SIGNATURE": signature,
            }
        )

    return headers


class TransactionClient(AsyncHTTPClient):
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
        api_creds: Optional[ClobCredentials] = None,
        builder_creds: Optional[BuilderConfig] = None,
        timeout: int = 30,
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
        assert api_creds is not None or builder_creds is not None, "Credentials missing"

        self.clob_client = ClobClient(api_creds.api_key)
        self.host = host
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.api_creds = api_creds
        self.builder_creds = builder_creds


    def set_api_credentials(self, api_creds: ClobCredentials) -> None:
        self.api_creds = api_creds

    async def derive_l2_api_credentials(
        self, signer: OrderSigner, nonce: int = 0
    ) -> ClobCredentials:
        """
        Derive L2 API credentials using L1 EIP-712 authentication via OrderSigner.

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

        # response = await self._request("POST", "/auth/api-key", headers=headers) # Alternative?
        response = await self._request("GET", "/auth/derive-api-key", headers=headers)

        # Debug: print response to see what we're getting
        print(f"[DEBUG] API Key Response: {response}")

        return ClobCredentials(
            api_key=response.get("apiKey", ""),
            secret=response.get("secret", ""),
            passphrase=response.get("passphrase", ""),
        )

    async def post_order(
        self, signed_order: Dict[str, Any], order_type: str = "GTC"
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
        body = {
            "order": signed_order.get("order", signed_order),
            "owner": self.funder.lower(),  # Polymarket expects lowercase owner
            "orderType": order_type,
        }

        # Add signature
        if "signature" in signed_order:
            body["signature"] = signed_order["signature"]

        body_json = json.dumps(body, separators=(",", ":"))

        # POST /order requires API credentials only (not builder creds)
        # Builder attribution is handled separately
        # IMPORTANT: Use the same address format that was used to derive credentials
        headers = create_headers(
            "POST",
            endpoint,
            body_json,
            funder=self.funder,  # Keep original case (checksummed)
            api_creds=self.api_creds,
            builder_creds=None,  # Don't send builder creds for order placement
        )

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
        body_json = json.dumps(body, separators=(",", ":"))
        headers = create_headers(
            "DELETE",
            endpoint,
            body_json,
            funder=self.funder,
            api_creds=self.api_creds,
            builder_creds=self.builder_creds,
        )

        return await self._request("DELETE", endpoint, data=body, headers=headers)

    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Cancellation response with canceled and not_canceled lists
        """
        endpoint = "/cancel-all"
        headers = create_headers(
            "DELETE",
            endpoint,
            funder=self.funder,
            api_creds=self.api_creds,
            builder_creds=self.builder_creds,
        )

        return await self._request("DELETE", endpoint, headers=headers)
