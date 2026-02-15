"""
Quick script to check what type of address is shown on Polymarket
"""
from web3 import Web3

# The address shown in Polymarket settings
polymarket_address = "0xF7F5C8669891e4c62dF672864E78a0B4B39757f8"

# The address from your config
config_address = "0x0177863ebd6b36b4cd0f667752fd825fd493e7ed"

# Connect to Polygon
rpc_url = "https://polygon-rpc.com"
w3 = Web3(Web3.HTTPProvider(rpc_url))

print("="*70)
print("ADDRESS ANALYSIS")
print("="*70)

def check_address(addr, label):
    print(f"\n{label}")
    print(f"Address: {addr}")

    checksum = Web3.to_checksum_address(addr)

    # Check if it's a contract
    code = w3.eth.get_code(checksum)

    if code == b'' or code == b'0x':
        print(f"Type: EOA (Externally Owned Account)")
        print(f"This is a regular wallet address")
    else:
        print(f"Type: CONTRACT")
        print(f"Bytecode size: {len(code)} bytes")

        # Try to determine contract type
        try:
            # Check for Gnosis Safe's getOwners() function
            owners_sig = Web3.keccak(text="getOwners()")[:4].hex()
            result = w3.eth.call({'to': checksum, 'data': owners_sig})

            from eth_abi import decode
            owners = decode(['address[]'], result)[0]

            print(f"\n✅ This is a GNOSIS SAFE")
            print(f"Number of owners: {len(owners)}")
            print(f"\nSafe Owners:")
            for i, owner in enumerate(owners, 1):
                print(f"  {i}. {owner}")

        except:
            print(f"\nℹ️  Might be a Polymarket Proxy or other contract")

    print("-"*70)

# Check both addresses
check_address(polymarket_address, "📍 ADDRESS FROM POLYMARKET SETTINGS")
check_address(config_address, "📍 ADDRESS FROM YOUR CONFIG")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\n1. The address shown in Polymarket settings is your TRADING address")
print("2. This is the address that should be in config.yaml as 'safe_address'")
print("3. Your private key should control this address (or be an owner of it)")
print("\nℹ️  Use the address from Polymarket settings: 0xF7F5C8669891e4c62dF672864E78a0B4B39757f8")
print("="*70)
