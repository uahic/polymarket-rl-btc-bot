"""
Diagnostic script to test L1 authentication with different signature types
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from config_loader import decrypt_private_key
from py_clob_client.signer import Signer
from py_clob_client.headers.headers import create_level_1_headers
from transactions.async_helpers import post, get


async def test_signature_type(sig_type: int, config: Config, private_key: str):
    """Test API key creation with a specific signature type"""
    print(f"\n{'='*60}")
    print(f"Testing signature type: {sig_type}")
    print(f"{'='*60}")

    signer = Signer(private_key=private_key, chain_id=config.clob.chain_id)

    print(f"Address: {signer.address()}")
    print(f"Chain ID: {config.clob.chain_id}")

    # Create L1 headers
    headers = create_level_1_headers(signer, nonce=0)
    print(f"\nL1 Headers:")
    for key, value in headers.items():
        if key == "POLY_SIGNATURE":
            print(f"  {key}: {value[:20]}...{value[-20:]}")
        else:
            print(f"  {key}: {value}")

    # Try DERIVE first (for existing accounts)
    print(f"\n[1] Trying DERIVE API key...")
    endpoint = f"{config.clob.host}/auth/derive-api-key"
    print(f"Endpoint: {endpoint}")

    try:
        response = await get(endpoint, headers=headers)
        print(f"\n✅ SUCCESS with DERIVE and signature_type={sig_type}")
        print(f"Response keys: {list(response.keys())}")
        return True
    except Exception as e:
        print(f"❌ DERIVE failed: {e}")

    # Try CREATE if derive failed
    print(f"\n[2] Trying CREATE API key...")
    endpoint = f"{config.clob.host}/auth/api-key"
    print(f"Endpoint: {endpoint}")

    try:
        response = await post(endpoint, headers=headers)
        print(f"\n✅ SUCCESS with CREATE and signature_type={sig_type}")
        print(f"Response keys: {list(response.keys())}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED with signature_type={sig_type}")
        print(f"Error: {e}")
        return False


async def main():
    """Test different signature types"""
    config = Config.load("config.yaml")
    private_key = decrypt_private_key()

    print("\nPolymarket L1 Authentication Diagnostic")
    print("="*60)
    print(f"Safe Address: {config.safe_address}")
    print(f"Current Config signature_type: {config.clob.signature_type}")
    print(f"Current Config tx_type: {config.relayer.tx_type}")

    # Test signature type 1 (POLY_PROXY)
    success_1 = await test_signature_type(1, config, private_key)

    # Test signature type 2 (GNOSIS_SAFE)
    success_2 = await test_signature_type(2, config, private_key)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Signature Type 1 (POLY_PROXY): {'✅ SUCCESS' if success_1 else '❌ FAILED'}")
    print(f"Signature Type 2 (GNOSIS_SAFE): {'✅ SUCCESS' if success_2 else '❌ FAILED'}")

    if success_1:
        print("\n📝 Recommendation: Use signature_type: 1 in config.yaml")
    elif success_2:
        print("\n📝 Recommendation: Use signature_type: 2 in config.yaml")
    else:
        print("\n⚠️  Both signature types failed. Possible issues:")
        print("   1. Private key doesn't match the safe_address")
        print("   2. Account not registered on Polymarket")
        print("   3. Network/API issues")


if __name__ == "__main__":
    asyncio.run(main())
