from app.clients.tessera_data import TesseraDataClient
from app.services.embedding import embed

TOOLS: list[dict] = [
    {
        "name": "get_user_history",
        "description": "Returns transaction history and behavioral signals for a user.",
        "input_schema": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "get_ip_risk",
        "description": "Returns IP reputation and risk score for an IP address.",
        "input_schema": {
            "type": "object",
            "properties": {"ip_address": {"type": "string"}},
            "required": ["ip_address"],
        },
    },
    {
        "name": "get_device_fingerprint",
        "description": (
            "Returns device fingerprint signals including suspicious flag and shared-user count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "check_blacklist",
        "description": "Checks if a user_id, email, or card_bin is on the fraud blacklist.",
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
        "description": "Searches historical fraud cases similar to the given transaction.",
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
        case _:
            return {"error": f"Unknown tool: {name}"}
