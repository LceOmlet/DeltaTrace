"""Fix the pinned AppWorld client's execution/HTTP deadline race in its owner.

The environment still interrupts generated code at its original timeout. Only
the HTTP read deadline for execute is removed, so the client can receive that
official result. Connection timeouts and all other RPCs remain unchanged. No
retry, synthetic observation, reward, or second execution is introduced.
"""
from pathlib import Path
import argparse


OLD = "                    timeout=_timeout_seconds,\n"
NEW = """                    # Code execution is bounded by the server's timeout_seconds.
                    # An identical HTTP read deadline races its timeout response.
                    # Preserve the connection deadline and consume the original
                    # execution result once; never retry a state-changing action.
                    timeout=(_timeout_seconds, None) if method_name == "execute" else _timeout_seconds,
"""


def patch_source(text: str) -> str:
    if NEW in text:
        return text
    if text.count(OLD) != 1:
        raise RuntimeError("Pinned AppWorld HTTP timeout site changed; inspect the owner source")
    return text.replace(OLD, NEW, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("appworld_root", type=Path)
    args = parser.parse_args()
    path = args.appworld_root / "src/appworld/environment.py"
    source = path.read_text()
    patched = patch_source(source)
    if patched != source:
        path.write_text(patched)
    print(f"AppWorld execute transport: {path} ({'patched' if patched != source else 'already patched'})")


if __name__ == "__main__":
    main()
