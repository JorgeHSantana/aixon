"""The knowledge base.

A real retriever would query a vector DB, a docs site, or a search API. This
one is a small in-memory list of FAQ entries so the example runs with zero
infrastructure. Each entry is just ``{"text": ..., "metadata": {...}}`` — the
shape `Retriever.search` must return.
"""

FAQ: list[dict] = [
    {
        "text": (
            "To reset your password, open the login page and click "
            "'Forgot password'. We email a reset link valid for 30 minutes."
        ),
        "metadata": {"topic": "account", "article": "KB-101"},
    },
    {
        "text": (
            "Acme Pro includes unlimited projects, priority support, and SSO. "
            "Upgrade any time from Settings -> Billing; the change is prorated."
        ),
        "metadata": {"topic": "billing", "article": "KB-204"},
    },
    {
        "text": (
            "We support SSO via SAML and OIDC on the Pro and Enterprise plans. "
            "Configure your identity provider under Settings -> Security."
        ),
        "metadata": {"topic": "security", "article": "KB-310"},
    },
    {
        "text": (
            "Data is encrypted at rest with AES-256 and in transit with TLS 1.3. "
            "Daily backups are retained for 30 days."
        ),
        "metadata": {"topic": "security", "article": "KB-311"},
    },
    {
        "text": (
            "The mobile app is available for iOS 16+ and Android 10+. "
            "Offline mode syncs automatically when you reconnect."
        ),
        "metadata": {"topic": "product", "article": "KB-420"},
    },
    {
        "text": (
            "To export your data, go to Settings -> Data and choose 'Export'. "
            "We generate a downloadable ZIP of CSV files within a few minutes."
        ),
        "metadata": {"topic": "product", "article": "KB-421"},
    },
]
