"""
Script to verify that private key matches the configured safe address
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from config_loader import decrypt_private_key
from py_clob_client.signer import Signer
from eth_account import Account
from web3 import Web3


def verify_key_match(config: Config, private_key: str):
    """Verify if the private key matches the safe address"""

    print("\n" + "="*70)
    print("PRIVATE KEY VERIFICATION")
    print("="*70)

    # 1. Get the EOA address from private key
    print("\n1️⃣  Deriving EOA address from private key...")
    try:
        # Use eth_account to derive the address from private key
        account = Account.from_key(private_key)
        eoa_address = account.address
        print(f"   EOA Address (from private key): {eoa_address}")
    except Exception as e:
        print(f"   ❌ ERROR: Could not derive address from private key: {e}")
        return False

    # 2. Get the configured safe address
    print("\n2️⃣  Configured safe address...")
    safe_address = Web3.to_checksum_address(config.safe_address)
    print(f"   Safe Address (from config):     {safe_address}")

    # 3. Compare addresses
    print("\n3️⃣  Comparing addresses...")
    eoa_checksum = Web3.to_checksum_address(eoa_address)

    if eoa_checksum == safe_address:
        print(f"   ✅ MATCH: The private key controls the safe address!")
        print(f"   → This is an EOA (Externally Owned Account)")
        print(f"   → You should use signature_type: 0 or 1")
        return True
    else:
        print(f"   ⚠️  NO MATCH: The addresses are different")
        print(f"\n   This means one of two things:")
        print(f"   a) The safe_address is a Gnosis Safe (multisig) contract")
        print(f"      → The private key is for one of the OWNERS of the Safe")
        print(f"      → You should use signature_type: 2 (GNOSIS_SAFE)")
        print(f"\n   b) The safe_address is a Polymarket Proxy contract")
        print(f"      → The private key is for the EOA that controls the proxy")
        print(f"      → You should use signature_type: 1 (POLY_PROXY)")
        print(f"\n   c) The private key is completely wrong ❌")
        print(f"      → You need to use the correct private key")
        return False

    # 4. Use py_clob_client signer to verify
    print("\n4️⃣  Verifying with py_clob_client Signer...")
    try:
        signer = Signer(private_key=private_key, chain_id=config.clob.chain_id)
        signer_address = signer.address()
        print(f"   Signer Address: {signer_address}")

        if Web3.to_checksum_address(signer_address) == eoa_checksum:
            print(f"   ✅ Signer address matches EOA address")
        else:
            print(f"   ❌ WARNING: Signer address doesn't match EOA address!")
    except Exception as e:
        print(f"   ❌ ERROR: Could not create signer: {e}")


def check_safe_type(safe_address: str, rpc_url: str):
    """Check if the address is a contract and what type"""

    print("\n" + "="*70)
    print("CHECKING ACCOUNT TYPE ON-CHAIN")
    print("="*70)

    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        if not w3.is_connected():
            print(f"❌ Could not connect to RPC: {rpc_url}")
            return

        print(f"\n✅ Connected to Polygon (Chain ID: {w3.eth.chain_id})")

        safe_checksum = Web3.to_checksum_address(safe_address)

        # Check if it's a contract
        code = w3.eth.get_code(safe_checksum)

        if code == b'' or code == b'0x':
            print(f"\n📝 Address Type: EOA (Externally Owned Account)")
            print(f"   → This is a regular wallet, not a contract")
            print(f"   → Recommended signature_type: 0 (EOA) or 1 (if using Polymarket proxy)")
        else:
            print(f"\n📝 Address Type: CONTRACT")
            print(f"   → Contract bytecode size: {len(code)} bytes")
            print(f"\n   Checking contract type...")

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

                print(f"   ✅ This appears to be a GNOSIS SAFE contract")
                print(f"   → Recommended signature_type: 2 (GNOSIS_SAFE)")

                # Try to decode owners
                try:
                    # Decode the ABI-encoded array
                    from eth_abi import decode
                    owners = decode(['address[]'], result)[0]
                    print(f"\n   Safe Owners ({len(owners)}):")
                    for i, owner in enumerate(owners, 1):
                        print(f"     {i}. {owner}")
                except:
                    pass

            except Exception as e:
                print(f"   ⚠️  Could not determine specific contract type")
                print(f"   → It might be a Polymarket Proxy or other contract")
                print(f"   → Try signature_type: 1 (POLY_PROXY) or 2 (GNOSIS_SAFE)")

    except Exception as e:
        print(f"\n❌ Error checking on-chain: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main verification function"""

    print("\n" + "="*70)
    print("POLYMARKET ACCOUNT VERIFICATION TOOL")
    print("="*70)

    # Load config
    config = Config.load("config.yaml")

    print(f"\nConfiguration:")
    print(f"  Safe Address:    {config.safe_address}")
    print(f"  Signature Type:  {config.clob.signature_type}")
    print(f"  Chain ID:        {config.clob.chain_id}")
    print(f"  RPC URL:         {config.rpc_url}")

    # Load private key
    try:
        private_key = decrypt_private_key()
        print(f"  Private Key:     Loaded ✓")
    except Exception as e:
        print(f"\n❌ ERROR: Could not load private key: {e}")
        return

    # Verify key matches
    verify_key_match(config, private_key)

    # Check on-chain account type
    check_safe_type(config.safe_address, config.rpc_url)

    # Summary
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\nBased on the checks above:")
    print("\n1. If EOA address matches safe address:")
    print("   → Use signature_type: 0 (EOA) or 1 (POLY_PROXY)")
    print("\n2. If safe address is a Gnosis Safe contract:")
    print("   → Use signature_type: 2 (GNOSIS_SAFE)")
    print("   → Make sure your private key is for one of the Safe owners")
    print("\n3. If safe address is a Polymarket Proxy contract:")
    print("   → Use signature_type: 1 (POLY_PROXY)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
