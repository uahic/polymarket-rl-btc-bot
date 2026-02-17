"""
Script to verify that private key matches the configured safe address
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from config_loader import decrypt_private_key
from py_clob_client.signer import Signer
from eth_account import Account
from web3 import Web3

logger = logging.getLogger(__name__)


def verify_key_match(config: Config, private_key: str):
    """Verify if the private key matches the safe address"""

    logger.info("\n" + "="*70)
    logger.info("PRIVATE KEY VERIFICATION")
    logger.info("="*70)

    # 1. Get the EOA address from private key
    logger.info("\n1️⃣  Deriving EOA address from private key...")
    try:
        # Use eth_account to derive the address from private key
        account = Account.from_key(private_key)
        eoa_address = account.address
        logger.info(f"   EOA Address (from private key): {eoa_address}")
    except Exception as e:
        logger.info(f"   ❌ ERROR: Could not derive address from private key: {e}")
        return False

    # 2. Get the configured safe address
    logger.info("\n2️⃣  Configured safe address...")
    safe_address = Web3.to_checksum_address(config.safe_address)
    logger.info(f"   Safe Address (from config):     {safe_address}")

    # 3. Compare addresses
    logger.info("\n3️⃣  Comparing addresses...")
    eoa_checksum = Web3.to_checksum_address(eoa_address)

    if eoa_checksum == safe_address:
        logger.info(f"   ✅ MATCH: The private key controls the safe address!")
        logger.info(f"   → This is an EOA (Externally Owned Account)")
        logger.info(f"   → You should use signature_type: 0 or 1")
        return True
    else:
        logger.info(f"   ⚠️  NO MATCH: The addresses are different")
        logger.info(f"\n   This means one of two things:")
        logger.info(f"   a) The safe_address is a Gnosis Safe (multisig) contract")
        logger.info(f"      → The private key is for one of the OWNERS of the Safe")
        logger.info(f"      → You should use signature_type: 2 (GNOSIS_SAFE)")
        logger.info(f"\n   b) The safe_address is a Polymarket Proxy contract")
        logger.info(f"      → The private key is for the EOA that controls the proxy")
        logger.info(f"      → You should use signature_type: 1 (POLY_PROXY)")
        logger.info(f"\n   c) The private key is completely wrong ❌")
        logger.info(f"      → You need to use the correct private key")
        return False

    # 4. Use py_clob_client signer to verify
    logger.info("\n4️⃣  Verifying with py_clob_client Signer...")
    try:
        signer = Signer(private_key=private_key, chain_id=config.clob.chain_id)
        signer_address = signer.address()
        logger.info(f"   Signer Address: {signer_address}")

        if Web3.to_checksum_address(signer_address) == eoa_checksum:
            logger.info(f"   ✅ Signer address matches EOA address")
        else:
            logger.info(f"   ❌ WARNING: Signer address doesn't match EOA address!")
    except Exception as e:
        logger.info(f"   ❌ ERROR: Could not create signer: {e}")


def check_safe_type(safe_address: str, rpc_url: str):
    """Check if the address is a contract and what type"""

    logger.info("\n" + "="*70)
    logger.info("CHECKING ACCOUNT TYPE ON-CHAIN")
    logger.info("="*70)

    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        if not w3.is_connected():
            logger.info(f"❌ Could not connect to RPC: {rpc_url}")
            return

        logger.info(f"\n✅ Connected to Polygon (Chain ID: {w3.eth.chain_id})")

        safe_checksum = Web3.to_checksum_address(safe_address)

        # Check if it's a contract
        code = w3.eth.get_code(safe_checksum)

        if code == b'' or code == b'0x':
            logger.info(f"\n📝 Address Type: EOA (Externally Owned Account)")
            logger.info(f"   → This is a regular wallet, not a contract")
            logger.info(f"   → Recommended signature_type: 0 (EOA) or 1 (if using Polymarket proxy)")
        else:
            logger.info(f"\n📝 Address Type: CONTRACT")
            logger.info(f"   → Contract bytecode size: {len(code)} bytes")
            logger.info(f"\n   Checking contract type...")

            # Try to determine if it's a Gnosis Safe
            # Gnosis Safe has specific function signatures
            try:
                # Check for Gnosis Safe's getOwners() function
                # Function signature: getOwners() -> 0xa0e67e2b
                owners_sig = Web3.keccak(text="getOwners()")[:4].hex()

                # Try to call it
                result = w3.eth.call({
                    'to': safe_checksum,
                    'data': owners_sig
                })

                logger.info(f"   ✅ This appears to be a GNOSIS SAFE contract")
                logger.info(f"   → Recommended signature_type: 2 (GNOSIS_SAFE)")

                # Try to decode owners
                try:
                    # Decode the ABI-encoded array
                    from eth_abi import decode
                    owners = decode(['address[]'], result)[0]
                    logger.info(f"\n   Safe Owners ({len(owners)}):")
                    for i, owner in enumerate(owners, 1):
                        logger.info(f"     {i}. {owner}")
                except:
                    pass

            except Exception as e:
                logger.info(f"   ⚠️  Could not determine specific contract type")
                logger.info(f"   → It might be a Polymarket Proxy or other contract")
                logger.info(f"   → Try signature_type: 1 (POLY_PROXY) or 2 (GNOSIS_SAFE)")

    except Exception as e:
        logger.info(f"\n❌ Error checking on-chain: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main verification function"""

    logger.info("\n" + "="*70)
    logger.info("POLYMARKET ACCOUNT VERIFICATION TOOL")
    logger.info("="*70)

    # Load config
    config = Config.load("config.yaml")

    logger.info(f"\nConfiguration:")
    logger.info(f"  Safe Address:    {config.safe_address}")
    logger.info(f"  Signature Type:  {config.clob.signature_type}")
    logger.info(f"  Chain ID:        {config.clob.chain_id}")
    logger.info(f"  RPC URL:         {config.rpc_url}")

    # Load private key
    try:
        private_key = decrypt_private_key()
        logger.info(f"  Private Key:     Loaded ✓")
    except Exception as e:
        logger.info(f"\n❌ ERROR: Could not load private key: {e}")
        return

    # Verify key matches
    verify_key_match(config, private_key)

    # Check on-chain account type
    check_safe_type(config.safe_address, config.rpc_url)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)
    logger.info("\nBased on the checks above:")
    logger.info("\n1. If EOA address matches safe address:")
    logger.info("   → Use signature_type: 0 (EOA) or 1 (POLY_PROXY)")
    logger.info("\n2. If safe address is a Gnosis Safe contract:")
    logger.info("   → Use signature_type: 2 (GNOSIS_SAFE)")
    logger.info("   → Make sure your private key is for one of the Safe owners")
    logger.info("\n3. If safe address is a Polymarket Proxy contract:")
    logger.info("   → Use signature_type: 1 (POLY_PROXY)")
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()
