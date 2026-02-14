import sys
import time
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import to_checksum_address, is_address

sys.path.insert(0, str(Path(__file__).parent.parent))
from structures.order import Order, OrderSide


class SignerError(Exception):
    """Base exception for signer operations."""

    pass


class OrderSigner:
    """
    Signs Polymarket orders using EIP-712.
    This is used to derive L2 credentials which can be used in the TransactionClient

    This signer handles:
    - Authentication messages (L1)
    - Order messages (for CLOB submission)

    Attributes:
        wallet: The Ethereum wallet instance
        address: The signer's address
        domain: EIP-712 domain separator
    """

    # Polymarket CLOB EIP-712 domain
    DOMAIN = {
        "name": "ClobAuthDomain",
        "version": "1",
        "chainId": 137,  # Polygon mainnet
    }

    # Order type definition for EIP-712
    ORDER_TYPES = {
        "Order": [
            {"name": "salt", "type": "uint256"},
            {"name": "maker", "type": "address"},
            {"name": "signer", "type": "address"},
            {"name": "taker", "type": "address"},
            {"name": "tokenId", "type": "uint256"},
            {"name": "makerAmount", "type": "uint256"},
            {"name": "takerAmount", "type": "uint256"},
            {"name": "expiration", "type": "uint256"},
            {"name": "nonce", "type": "uint256"},
            {"name": "feeRateBps", "type": "uint256"},
            {"name": "side", "type": "uint8"},
            {"name": "signatureType", "type": "uint8"},
        ]
    }

    def __init__(self, private_key: str):
        """
        Initialize signer with a private key.

        Args:
            private_key: Private key (with or without 0x prefix)

        Raises:
            ValueError: If private key is invalid
        """
        if private_key.startswith("0x"):
            private_key = private_key[2:]

        try:
            self.wallet = Account.from_key(f"0x{private_key}")
        except Exception as e:
            raise ValueError(f"Invalid private key: {e}")

        self.address = self.wallet.address

    def sign_auth_message(self, timestamp: Optional[str] = None, nonce: int = 0) -> str:
        """
        Sign an authentication message for L1 authentication.

        This signature is used to create or derive API credentials.

        Args:
            timestamp: Message timestamp (defaults to current time)
            nonce: Message nonce (usually 0)

        Returns:
            Hex-encoded signature
        """
        if timestamp is None:
            timestamp = str(int(time.time()))

        # Auth message types
        auth_types = {
            "ClobAuth": [
                {"name": "address", "type": "address"},
                {"name": "timestamp", "type": "string"},
                {"name": "nonce", "type": "uint256"},
                {"name": "message", "type": "string"},
            ]
        }

        message_data = {
            "address": self.address,
            "timestamp": timestamp,
            "nonce": nonce,
            "message": "This message attests that I control the given wallet",
        }

        signable = encode_typed_data(
            domain_data=self.DOMAIN, message_types=auth_types, message_data=message_data
        )

        signed = self.wallet.sign_message(signable)
        return "0x" + signed.signature.hex()

    def sign_order(
        self,
        order: Order,
        expiration: Optional[int] = None,
        taker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sign a Polymarket order.

        Args:
            order: Order instance to sign
            expiration: Order expiration timestamp (0 = no expiration, default: 1 second from now)
            taker: Specific taker address (default: zero address for public orders)

        Returns:
            Dictionary containing order and signature

        Raises:
            SignerError: If signing fails
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(order.token_id, str) or not order.token_id:
                raise ValueError(f"Invalid token_id: {order.token_id}")

            try:
                token_id_int = int(order.token_id)
                if token_id_int < 0:
                    raise ValueError(f"token_id must be non-negative: {token_id_int}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"token_id must be numeric: {order.token_id}") from e

            try:
                maker_amount_int = int(order.maker_amount)
                taker_amount_int = int(order.taker_amount)
                if maker_amount_int <= 0 or taker_amount_int <= 0:
                    raise ValueError(
                        f"Amounts must be positive: maker={maker_amount_int}, taker={taker_amount_int}"
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Order amounts must be numeric") from e

            # Set default expiration to 30 seconds from now if not specified
            if expiration is None:
                expiration = int(time.time()) + 30

            # Set default taker to zero address if not specified
            if taker is None:
                taker = "0x0000000000000000000000000000000000000000"
            else:
                if not is_address(taker):
                    raise ValueError(f"Invalid taker address: {taker}")
                taker = to_checksum_address(taker)

            # Generate cryptographically secure random salt as uint256
            salt = secrets.randbits(256)

            # Build order message for EIP-712
            order_message = {
                "salt": salt,
                "maker": to_checksum_address(order.maker),
                "signer": self.address,
                "taker": taker,
                "tokenId": token_id_int,
                "makerAmount": maker_amount_int,
                "takerAmount": taker_amount_int,
                "expiration": expiration,
                "nonce": order.nonce,
                "feeRateBps": order.fee_rate_bps,
                "side": order.side_value,
                "signatureType": order.signature_type,
            }

            # Sign the order using new API format
            signable = encode_typed_data(
                domain_data=self.DOMAIN,
                message_types=self.ORDER_TYPES,
                message_data=order_message,
            )

            signed = self.wallet.sign_message(signable)

            return {
                "order": {
                    "salt": salt,
                    "maker": to_checksum_address(order.maker),
                    "signer": self.address,
                    "taker": taker,
                    "tokenId": order.token_id,
                    "makerAmount": order.maker_amount,
                    "takerAmount": order.taker_amount,
                    "expiration": str(expiration),
                    "nonce": str(order.nonce),
                    "feeRateBps": str(order.fee_rate_bps),
                    "side": str(order.side_value),
                    "signatureType": str(order.signature_type),
                },
                "signature": "0x" + signed.signature.hex(),
            }

        except SignerError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise SignerError(f"Failed to sign order: {e}") from e

    def sign_order_dict(
        self,
        token_id: str,
        price: float,
        size: float,
        side: OrderSide,
        maker: str,
        nonce: Optional[int] = None,
        fee_rate_bps: int = 0,
        expiration: Optional[int] = None,
        taker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sign an order from dictionary parameters.

        Args:
            token_id: Market token ID
            price: Price per share (0-1 for binary markets)
            size: Number of shares (must be positive)
            side: OrderSide.BUY or OrderSide.SELL
            maker: Maker's wallet address
            nonce: Order nonce (defaults to timestamp)
            fee_rate_bps: Fee rate in basis points (0-10000)
            expiration: Order expiration timestamp (default: 1 second from now)
            taker: Specific taker address (default: zero address for public orders)

        Returns:
            Dictionary containing order and signature

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs before creating Order
        if not isinstance(price, (int, float)):
            raise ValueError(f"price must be numeric, got {type(price).__name__}")
        if not (0 < price <= 1):
            raise ValueError(f"price must be between 0 and 1, got {price}")

        if not isinstance(size, (int, float)):
            raise ValueError(f"size must be numeric, got {type(size).__name__}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        if not isinstance(side, OrderSide):
            raise ValueError(f"side must be OrderSide enum, got {type(side).__name__}")

        if not isinstance(maker, str) or not is_address(maker):
            raise ValueError(f"maker must be a valid Ethereum address, got {maker}")

        if not isinstance(fee_rate_bps, int) or not (0 <= fee_rate_bps <= 10000):
            raise ValueError(
                f"fee_rate_bps must be an integer between 0-10000, got {fee_rate_bps}"
            )

        order = Order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            maker=maker,
            nonce=nonce,
            fee_rate_bps=fee_rate_bps,
        )
        return self.sign_order(order, expiration=expiration, taker=taker)
