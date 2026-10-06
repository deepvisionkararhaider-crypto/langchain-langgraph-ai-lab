"""Project 05 — Customer Support Agent: a real (curated) support knowledge base.

A small but genuine SaaS help-center: articles across the categories a support
agent must handle (billing, account, technical, shipping, refunds, security,
integrations, plans). Retrieval over this KB grounds every generated answer.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    tags: tuple[str, ...]
    content: str


KB: list[Article] = [
    Article(
        id="KB-001",
        title="How to change your subscription plan",
        category="plans",
        tags=("plan", "upgrade", "downgrade", "subscription", "change"),
        content=(
            "You can change your plan at any time from Settings → Billing → Change plan. "
            "Upgrades take effect immediately and are prorated for the remainder of the current "
            "billing cycle. Downgrades take effect at the start of the next billing cycle, and you "
            "keep access to your current plan's features until then. The Starter plan includes 5 "
            "seats; Pro includes unlimited seats. Enterprise plans require contacting sales."
        ),
    ),
    Article(
        id="KB-002",
        title="Billing cycles, invoices and payment methods",
        category="billing",
        tags=("billing", "invoice", "payment", "credit card", "receipt", "vat"),
        content=(
            "Invoices are issued on the first day of each billing cycle and are available under "
            "Settings → Billing → Invoices (PDF). We accept major credit cards and SEPA direct "
            "debit. To update a payment method, go to Settings → Billing → Payment methods. "
            "VAT/GST numbers can be added in the same screen; they will appear on future invoices. "
            "Failed payments are retried 3 times over 7 days before the account is set to read-only."
        ),
    ),
    Article(
        id="KB-003",
        title="Refund policy",
        category="refunds",
        tags=("refund", "money back", "cancel", "charge", "duplicate"),
        content=(
            "We offer a 30-day money-back guarantee on first-time purchases of annual plans. "
            "Monthly plans are non-refundable but can be cancelled to stop future charges. "
            "Duplicate or accidental charges are refunded in full within 5 business days. "
            "To request a refund, open a ticket with your invoice number and the reason; "
            "approved refunds are returned to the original payment method within 5-10 business days."
        ),
    ),
    Article(
        id="KB-004",
        title="Resetting your password and enabling 2FA",
        category="account",
        tags=("password", "reset", "login", "2fa", "mfa", "security", "locked out"),
        content=(
            "Click 'Forgot password' on the sign-in page and follow the email link (valid for 60 "
            "minutes). If you don't receive it, check spam or ask an admin to resend it. Two-factor "
            "authentication (2FA) can be enabled under Settings → Security → Two-factor. We "
            "recommend an authenticator app; SMS is also supported. If you are locked out and have "
            "lost your 2FA device, use a recovery code or contact support to verify your identity."
        ),
    ),
    Article(
        id="KB-005",
        title="Account deletion and data export",
        category="account",
        tags=("delete account", "gdpr", "export", "data", "privacy", "close account"),
        content=(
            "You can export all of your data under Settings → Data → Export (JSON or CSV). "
            "To delete your account and all associated data, go to Settings → Account → Delete "
            "account. Deletion is immediate and irreversible after a 7-day grace period, during "
            "which you can restore the account by signing back in. GDPR requests are processed "
            "within 30 days."
        ),
    ),
    Article(
        id="KB-006",
        title="API rate limits and authentication",
        category="technical",
        tags=("api", "rate limit", "429", "token", "authentication", "key", "sdks"),
        content=(
            "Authenticate API calls with a bearer token created under Settings → API → Tokens. "
            "Rate limits are 100 requests/minute per token on Starter and 1000 requests/minute on "
            "Pro; exceeding them returns HTTP 429 with a Retry-After header. Implement exponential "
            "backoff. Tokens are shown only once at creation; rotate tokens regularly and never "
            "embed them in client-side code."
        ),
    ),
    Article(
        id="KB-007",
        title="Troubleshooting: dashboard not loading / 500 errors",
        category="technical",
        tags=("error", "not loading", "500", "bug", "broken", "dashboard", "outage"),
        content=(
            "If the dashboard fails to load: (1) hard-refresh the browser and clear cache; "
            "(2) check status.acmecloud.example for active incidents; (3) try an incognito window "
            "to rule out extensions; (4) confirm your network allows websockets. If errors persist, "
            "open a ticket with the time, your region, and the request ID shown in the error banner. "
            "Our on-call team triages P1 incidents within 15 minutes."
        ),
    ),
    Article(
        id="KB-008",
        title="SLA, uptime and support response targets",
        category="plans",
        tags=("sla", "uptime", "support", "response time", "priority", "99.9"),
        content=(
            "All paid plans include email support with a 24-hour first-response target. Pro and "
            "Enterprise customers get priority Slack Connect support with a 4-hour first-response "
            "target. Pro includes a 99.9% uptime SLA; Enterprise can negotiate 99.99%. Service "
            "credits are available if monthly uptime falls below the SLA, on request."
        ),
    ),
    Article(
        id="KB-009",
        title="Shipping and delivery for hardware orders",
        category="shipping",
        tags=("shipping", "delivery", "tracking", "hardware", "order", "late"),
        content=(
            "Hardware orders ship within 2 business days; you'll receive a tracking link by email. "
            "Standard delivery is 3-5 business days (EU/US) and 5-10 days elsewhere. If an order is "
            "more than 5 days late, contact support with your order number for a trace and a "
            "replacement if the carrier confirms loss. Accessories are non-returnable once opened."
        ),
    ),
    Article(
        id="KB-010",
        title="Integrations: Slack, webhooks and SSO",
        category="integrations",
        tags=("integration", "slack", "webhook", "sso", "saml", "okta", "connect"),
        content=(
            "Connect Slack under Settings → Integrations → Slack (OAuth, no token handling needed). "
            "Webhooks can be configured per event with a signing secret; verify signatures on your "
            "endpoint. SSO via SAML is available on Pro and Enterprise and supports Okta, Azure AD, "
            "and Google Workspace. SCIM user provisioning is Enterprise-only."
        ),
    ),
    Article(
        id="KB-011",
        title="Data security, encryption and compliance",
        category="security",
        tags=("security", "encryption", "aes", "tls", "soc2", "compliance", "iso"),
        content=(
            "All data is encrypted at rest with AES-256 and in transit with TLS 1.3. We are SOC 2 "
            "Type II certified and GDPR compliant. Data is stored in Frankfurt (eu-central) by "
            "default; US-East residency is available on request. Backups are retained 30 days on "
            "Starter and 365 days on Pro/Enterprise. Penetration test summaries are available "
            "under NDA."
        ),
    ),
    Article(
        id="KB-012",
        title="Team seats, roles and permissions",
        category="account",
        tags=("seats", "team", "members", "roles", "permissions", "invite", "admin"),
        content=(
            "Invite teammates under Settings → Team → Invite. Roles are Owner, Admin, Editor and "
            "Viewer. Owners and Admins can manage billing and members; Editors can create content; "
            "Viewers have read-only access. Starter includes 5 seats; Pro and Enterprise are "
            "unlimited. Removing a member frees the seat immediately."
        ),
    ),
]


def all_articles() -> list[Article]:
    return KB


def categories() -> list[str]:
    return sorted({a.category for a in KB})


def by_id(article_id: str) -> Article | None:
    return next((a for a in KB if a.id == article_id), None)
