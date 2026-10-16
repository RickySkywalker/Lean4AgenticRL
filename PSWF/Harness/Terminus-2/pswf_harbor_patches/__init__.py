"""Harbor patches and fixed source identities for the PSWF release.

The tokenizer and summarizer patches supply the agent's context accounting.
The other patches configure container cleanup, networking, hooks, and verifier
dependency retries. The CLI validates the pinned upstream source and the
resulting release artifacts."""

HARBOR_REPO = "https://github.com/harbor-framework/harbor.git"
HARBOR_REV = "f03db62fd2ed2ed1f79aefe024cfcbc68a0d759e"
HARBOR_VERSION = "0.21.0"

# (number of harbor/**/*.py files, sha256 of the canonical [{path, sha256}] listing) -- see __main__.tree_digest
PRISTINE_TREE = (308, "da69af1a70e8e9f0a03f1d431ecc32e083f5a97bc5addad8068e2758d3a58477")
PATCHED_TREE = (308, 'ebd81c4a0b3f1aa5eb16040aef0beb5c422bab4c30f01ca099d6607eee9b6363')

# Release identity of the agent package (see __main__.harness_identity).
HARNESS_IDENTITY = '1f3258783e599809da8565b81b127e16e27239d4e6e7d0929e088843b630b7a8'

__version__ = "1.0.0"
