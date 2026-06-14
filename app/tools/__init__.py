from app.clients.tessera_data import TesseraDataClient
from app.services.embedding import embed

TOOLS: list[dict] = [
    {
        "name": "get_user_history",
        "description": (
            "Retrieves the complete transaction history and chargeback record"
            " for a user account. "
            "Returns: total_transactions, chargeback_count, chargeback_rate,"
            " avg_transaction_amount, account_age_days, last_transaction_at. "
            "Call this FIRST for every analysis — it establishes the user's"
            " baseline. A new account (account_age_days < 30,"
            " total_transactions < 5) with no history is a risk amplifier for"
            " all other signals. An established account with zero chargebacks"
            " provides strong prior toward APPROVE. "
            "Do not call this more than once per analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_ip_risk",
        "description": (
            "Queries the IP reputation service for the originating IP address. "
            "Returns: risk_score (0.0-1.0), vpn_detected, proxy_detected,"
            " tor_detected, country_code, isp, abuse_reports_30d. "
            "Call this early in every analysis — IP signals often determine"
            " whether to escalate velocity checks. A risk_score > 0.8 or"
            " tor_detected=true combined with a new account is a near-certain"
            " DECLINE signal. "
            "Do not rely on this as the sole basis for a DECLINE; always"
            " combine with at least one other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ip_address": {"type": "string"}},
            "required": ["ip_address"],
        },
    },
    {
        "name": "get_device_fingerprint",
        "description": (
            "Looks up the device fingerprint to determine whether this device"
            " has been seen before and whether it has a fraud history. "
            "Returns: device_seen_before, transaction_count_on_device,"
            " fraud_flag, device_type, os, browser, first_seen_at. "
            "Call this after get_user_history — a mismatch between an"
            " established user and a brand-new device is a key fraud indicator. "
            "Do not call this if device_id is not present in the transaction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "check_blacklist",
        "description": (
            "Checks whether the user ID, email address, or card BIN appears on"
            " any internal or shared fraud blacklist. "
            "Returns: matched (bool), match_type (user|email|card_bin|none),"
            " matched_field, blacklist_source, blacklist_added_at. "
            "Call this for every transaction — a blacklist match is grounds for"
            " immediate DECLINE without needing additional signals. If"
            " matched=true, cite this as a source in cited_sources. "
            "Do not skip this step even when the user history looks clean."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "email": {"type": "string"},
                "card_bin": {"type": "string"},
            },
            "required": ["user_id", "email", "card_bin"],
        },
    },
    {
        "name": "search_similar_cases",
        "description": (
            "Performs a semantic similarity search over the historical case"
            " database (RAG corpus) to find past transactions that resemble"
            " the current one. "
            "Returns: up to 5 similar cases, each with case_id, decision,"
            " similarity_score, key_signals. "
            "Call this when signals are conflicting or ambiguous — retrieved"
            " cases provide grounded precedent that satisfies the grounding"
            " requirement. Always cite the case_id of any retrieved case you"
            " use in your reasoning. "
            "Do not call this when the decision is already clear from blacklist"
            " or velocity signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "amount": {"type": "number"},
                "merchant_category": {"type": "string"},
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "check_velocity",
        "description": (
            "Check cross-user velocity for an IP address or card BIN within a recent time"
            " window. Returns the number of distinct user accounts that have transacted from"
            " this IP address or this card BIN in the last N minutes. A high count of distinct"
            " users on a single IP or BIN is the strongest signal of card-testing attacks —"
            " coordinated fraud attempts where attackers cycle through many stolen cards from"
            " one location. Call this when the transaction involves a new account or a new"
            " device, or when the IP risk score is elevated, BEFORE deciding to escalate."
            " Do not call this when the user has a long established history with no chargeback"
            " flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ip_address": {
                    "type": "string",
                    "description": "The IP address from the transaction.",
                },
                "card_bin": {
                    "type": "string",
                    "description": "The first 6-8 digits of the card number (BIN).",
                },
                "window_minutes": {
                    "type": "integer",
                    "description": "Look-back window in minutes. Default 60.",
                    "default": 60,
                },
            },
            "required": ["ip_address", "card_bin"],
        },
    },
]


async def dispatch(
    name: str,
    args: dict,
    client: TesseraDataClient,
    trace_id: str | None = None,
) -> dict:
    match name:
        case "get_user_history":
            return await client.get_user_history(args["user_id"], trace_id=trace_id)
        case "get_ip_risk":
            return await client.get_ip_risk(args["ip_address"], trace_id=trace_id)
        case "get_device_fingerprint":
            return await client.get_device_fingerprint(args["device_id"], trace_id=trace_id)
        case "check_blacklist":
            return await client.check_blacklist(
                args["user_id"],
                args["email"],
                args["card_bin"],
                trace_id=trace_id,
            )
        case "search_similar_cases":
            text = f"{args.get('merchant_category', '')} {args.get('amount', '')}"
            embedding = await embed(text)
            return await client.search_similar_cases(embedding, limit=5, trace_id=trace_id)
        case "check_velocity":
            return await client.check_velocity(
                ip_address=args["ip_address"],
                card_bin=args["card_bin"],
                window_minutes=args.get("window_minutes", 60),
                trace_id=trace_id,
            )
        case _:
            return {"error": f"Unknown tool: {name}"}
