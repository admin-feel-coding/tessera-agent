from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import anthropic
import structlog
from anthropic import APIStatusError

from app.clients.tessera_data import TesseraDataClient
from app.config import settings
from app.observability import langfuse, start_trace
from app.schemas.transaction import Transaction
from app.schemas.verdict import (
    CitedSource,
    Decision,
    EscalationCategory,
    Signals,
    SourceType,
    SSEEvent,
    Verdict,
)
from app.services.embedding import embed
from app.services.pricing import cost_usd as compute_cost_usd
from app.tools import TOOLS, dispatch

log = structlog.get_logger(__name__)

IS_MOCK = settings.anthropic_api_key in ("", "placeholder-replace-me")


def _extract_json(text: str) -> dict:
    """Extract a JSON object from Claude's response, which may be wrapped in markdown."""
    import re

    text = text.strip()
    # 1. Try plain JSON first (the prompt asks for this)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Try ```json ... ``` code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Try the outermost { ... } in the string
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON object found in response (first 200 chars): {text[:200]}")


_SYSTEM_PROMPT = (
    "You are Tessera, an expert AI fraud analyst. "
    "Your job is to analyze a financial transaction and return a structured verdict.\n\n"
    "You have access to tools that query live data signals. "
    "Call the tools relevant to the transaction fields provided.\n\n"
    "After gathering evidence, return ONLY a JSON object "
    "(no markdown, no explanation outside the JSON) with this exact shape:\n"
    "{\n"
    '  "decision": "APPROVE | DECLINE | ESCALATE",\n'
    '  "risk_score": <float 0.0-1.0>,\n'
    '  "reasoning": "<step-by-step explanation>",\n'
    '  "cited_sources": [\n'
    '    {"type": "rule | case", "id": "<rule name or case ID>",\n'
    '     "excerpt": "<the specific part supporting your verdict>"}\n'
    "  ],\n"
    '  "signals": {\n'
    '    "user_history_flag": <bool>,\n'
    '    "ip_risk_flag": <bool>,\n'
    '    "device_fingerprint_flag": <bool>,\n'
    '    "blacklist_hit": <bool>,\n'
    '    "velocity_flag": <bool>\n'
    "  },\n"
    '  "escalation_reason": "<string or null>",\n'
    '  "escalation_category": "<CONFLICTING_SIGNALS | INSUFFICIENT_GROUNDING'
    ' | LOW_CONFIDENCE | NOVEL_PATTERN | POLICY_REQUIRED | null>"\n'
    "}\n\n"
    "Rules:\n"
    "- cited_sources must have at least one entry. "
    "If you cannot cite a rule or retrieved case, set decision to ESCALATE.\n"
    "- Base every verdict on evidence from tool results only "
    "— never on general knowledge alone.\n"
    "- DECLINE only when you have a direct blacklist hit or a highly specific rule match.\n"
    "- ESCALATE when evidence is ambiguous or insufficient for a confident verdict.\n"
    "- escalation_category: one of CONFLICTING_SIGNALS, INSUFFICIENT_GROUNDING, "
    "LOW_CONFIDENCE, NOVEL_PATTERN, POLICY_REQUIRED"
    " — required when decision is ESCALATE, null otherwise.\n"
)


async def _run_mock(transaction: Transaction, trace: object) -> Verdict:
    start = time.monotonic()

    data_client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)
    trace_id: str = getattr(trace, "id", "")
    try:
        history, ip_risk, device, blacklist, similar, velocity = await _gather_all_signals(
            transaction, data_client, trace, trace_id
        )
    finally:
        await data_client.close()

    signals = _derive_signals(blacklist, ip_risk, device, history, velocity)

    blacklist_hit = signals.blacklist_hit
    flag_count = sum(
        [
            signals.ip_risk_flag,
            signals.device_fingerprint_flag,
            signals.user_history_flag,
            signals.velocity_flag,
        ]
    )

    if blacklist_hit:
        decision = Decision.DECLINE
        risk_score = 0.95
        cited_sources = [
            CitedSource(
                type=SourceType.RULE,
                id="BLACKLIST_HIT",
                excerpt="Entity matched fraud blacklist.",
            )
        ]
        escalation_reason = None
        escalation_category = None
    elif flag_count >= 2:
        decision = Decision.ESCALATE
        risk_score = 0.65
        cited_sources = [
            CitedSource(
                type=SourceType.RULE,
                id="MULTI_SIGNAL_RISK",
                excerpt="Multiple risk signals present; escalating for human review.",
            )
        ]
        escalation_reason = "Multiple risk signals detected; human review required."
        escalation_category = EscalationCategory.LOW_CONFIDENCE
    else:
        decision = Decision.APPROVE
        risk_score = 0.10
        cited_sources = [
            CitedSource(
                type=SourceType.RULE,
                id="NO_RISK_FLAGS",
                excerpt="No risk signals detected across all checks.",
            )
        ]
        escalation_reason = None
        escalation_category = None

    # If a similar case was retrieved, add it as an additional cited source
    cases = similar.get("cases", [])
    if cases:
        top = cases[0]
        cited_sources.append(
            CitedSource(
                type=SourceType.CASE,
                id=top.get("id", ""),
                excerpt=(top.get("reasoning", "") or "")[:200],
            )
        )

    reasoning = (
        f"User history: {history.get('transaction_count', 0)} transactions, "
        f"high_velocity={history.get('high_velocity', False)}. "
        f"IP risk score: {ip_risk.get('risk_score', 0.0)}, is_vpn={ip_risk.get('is_vpn', False)}. "
        f"Device suspicious={device.get('suspicious', False)}, "
        f"shared_users={device.get('user_count', 1)}. "
        f"Blacklist match={blacklist.get('match', False)}."
    )

    latency_ms = int((time.monotonic() - start) * 1000)

    log.info(
        "analyze_complete",
        runner="mock",
        transaction_id=transaction.transaction_id,
        decision=decision,
        latency_ms=latency_ms,
    )

    tool_calls_count = 6  # 5 signal calls + 1 similarity search
    trace.update(
        output={
            "decision": decision,
            "risk_score": risk_score,
            "cited_sources_count": len(cited_sources),
            "tool_calls": tool_calls_count,
        },
        metadata={"escalated": decision == Decision.ESCALATE},
    )

    return Verdict.model_validate(
        {
            "transaction_id": transaction.transaction_id,
            "decision": decision,
            "risk_score": risk_score,
            "reasoning": reasoning,
            "cited_sources": [s.model_dump() for s in cited_sources],
            "signals": signals.model_dump(),
            "escalation_reason": escalation_reason,
            "escalation_category": escalation_category,
            "latency_ms": latency_ms,
            "model": "mock-runner-v1",
            "tool_calls": tool_calls_count,
            "langfuse_trace_id": trace_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "ip_address": transaction.ip_address,
            "card_bin": transaction.card_bin,
            "device_id": transaction.device_id,
        }
    )


async def _run_real(transaction: Transaction, trace: object) -> Verdict:
    start = time.monotonic()
    trace_id: str = getattr(trace, "id", "")

    ai_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    data_client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)

    messages: list[dict] = [
        {
            "role": "user",
            "content": f"Analyze this transaction:\n{transaction.model_dump_json(indent=2)}",
        }
    ]

    tool_call_count = 0
    final_text: str | None = None
    total_in = 0
    total_out = 0

    try:
        while True:
            generation = trace.generation(
                name="claude.completion",
                model="claude-sonnet-4-6",
                input=messages,
            )
            for attempt in range(settings.max_analyze_retries + 1):
                try:
                    response = await ai_client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=2048,
                        system=_SYSTEM_PROMPT,
                        tools=TOOLS,  # type: ignore[arg-type]
                        messages=messages,  # type: ignore[arg-type]
                    )
                    break
                except APIStatusError as e:
                    if e.status_code in {429, 529} and attempt < settings.max_analyze_retries:
                        wait = 2**attempt
                        log.warn(
                            "anthropic_retryable_error",
                            status=e.status_code,
                            attempt=attempt,
                            wait_s=wait,
                            transaction_id=transaction.transaction_id,
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise
            generation.end(
                output=response.content,
                usage={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                        break
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if tool_call_count >= 10:
                            log.warn(
                                "max_tool_calls_exceeded",
                                transaction_id=transaction.transaction_id,
                            )
                            final_text = None
                            break
                        span = trace.span(
                            name=f"tool.{block.name}",
                            input=block.input,
                        )
                        result = await dispatch(
                            block.name,
                            block.input,
                            data_client,
                            trace_id=trace_id,
                        )
                        span.end(output=result)
                        tool_call_count += 1
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result),
                            }
                        )

                if final_text is not None or tool_call_count >= 10:
                    break

                messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]
                messages.append({"role": "user", "content": tool_results})
                continue

            break

    finally:
        await data_client.close()
        langfuse.flush()

    latency_ms = int((time.monotonic() - start) * 1000)

    if tool_call_count >= 10 or final_text is None:
        log.warn(
            "analyze_escalated_no_output",
            transaction_id=transaction.transaction_id,
            tool_calls=tool_call_count,
        )
        _msg = "Agent reached maximum tool call limit without producing a verdict."
        _escalate_verdict = {
            "transaction_id": transaction.transaction_id,
            "decision": Decision.ESCALATE,
            "risk_score": 0.5,
            "reasoning": _msg,
            "cited_sources": [],
            "signals": {
                "user_history_flag": False,
                "ip_risk_flag": False,
                "device_fingerprint_flag": False,
                "blacklist_hit": False,
                "velocity_flag": False,
            },
            "escalation_reason": _msg,
            "latency_ms": latency_ms,
            "model": "claude-sonnet-4-6",
            "tool_calls": tool_call_count,
            "langfuse_trace_id": trace_id,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "ip_address": transaction.ip_address,
            "card_bin": transaction.card_bin,
            "device_id": transaction.device_id,
        }
        trace.update(
            output={
                "decision": Decision.ESCALATE,
                "risk_score": 0.5,
                "cited_sources_count": 0,
                "tool_calls": tool_call_count,
            },
            metadata={"escalated": True},
        )
        return Verdict.model_validate(_escalate_verdict)

    try:
        raw = _extract_json(final_text)
    except (json.JSONDecodeError, ValueError):
        log.warn(
            "agent_non_parseable_output",
            transaction_id=transaction.transaction_id,
            raw_text=final_text[:200],
        )
        _parse_fail = {
            "transaction_id": transaction.transaction_id,
            "decision": Decision.ESCALATE,
            "risk_score": 0.5,
            "reasoning": final_text,
            "cited_sources": [],
            "signals": {
                "user_history_flag": False,
                "ip_risk_flag": False,
                "device_fingerprint_flag": False,
                "blacklist_hit": False,
                "velocity_flag": False,
            },
            "escalation_reason": "Agent produced non-parseable output.",
            "latency_ms": latency_ms,
            "model": "claude-sonnet-4-6",
            "tool_calls": tool_call_count,
            "langfuse_trace_id": trace_id,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "ip_address": transaction.ip_address,
            "card_bin": transaction.card_bin,
            "device_id": transaction.device_id,
        }
        trace.update(
            output={
                "decision": Decision.ESCALATE,
                "risk_score": 0.5,
                "cited_sources_count": 0,
                "tool_calls": tool_call_count,
            },
            metadata={"escalated": True},
        )
        return Verdict.model_validate(_parse_fail)

    cited_sources_raw = raw.get("cited_sources", [])
    cited_sources = [CitedSource.model_validate(s) for s in cited_sources_raw]
    signals_raw = raw.get(
        "signals",
        {
            "user_history_flag": False,
            "ip_risk_flag": False,
            "device_fingerprint_flag": False,
            "blacklist_hit": False,
            "velocity_flag": False,
        },
    )
    decision = raw.get("decision", Decision.ESCALATE)

    log.info(
        "analyze_complete",
        runner="real",
        transaction_id=transaction.transaction_id,
        decision=decision,
        latency_ms=latency_ms,
        tool_calls=tool_call_count,
    )

    trace.update(
        output={
            "decision": decision,
            "risk_score": float(raw.get("risk_score", 0.5)),
            "cited_sources_count": len(cited_sources),
            "tool_calls": tool_call_count,
        },
        metadata={"escalated": decision == Decision.ESCALATE},
    )

    return Verdict.model_validate(
        {
            "transaction_id": transaction.transaction_id,
            "decision": decision,
            "risk_score": float(raw.get("risk_score", 0.5)),
            "reasoning": raw.get("reasoning", ""),
            "cited_sources": [s.model_dump() for s in cited_sources],
            "signals": signals_raw,
            "escalation_reason": raw.get("escalation_reason"),
            "escalation_category": raw.get("escalation_category"),
            "latency_ms": latency_ms,
            "model": "claude-sonnet-4-6",
            "tool_calls": tool_call_count,
            "langfuse_trace_id": trace_id,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "ip_address": transaction.ip_address,
            "card_bin": transaction.card_bin,
            "device_id": transaction.device_id,
        }
    )


async def _gather_all_signals(
    transaction: Transaction,
    client: TesseraDataClient,
    trace: object,
    trace_id: str,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    span = trace.span(name="tool.get_user_history", input={"user_id": transaction.user_id})
    history = await client.get_user_history(transaction.user_id, trace_id=trace_id)
    span.end(output=history)

    if transaction.ip_address:
        span = trace.span(name="tool.get_ip_risk", input={"ip_address": transaction.ip_address})
        ip_risk = await client.get_ip_risk(transaction.ip_address, trace_id=trace_id)
        span.end(output=ip_risk)
    else:
        ip_risk = {"risk_score": 0.0, "is_vpn": False, "country": "unknown"}

    if transaction.device_id:
        span = trace.span(
            name="tool.get_device_fingerprint", input={"device_id": transaction.device_id}
        )
        device = await client.get_device_fingerprint(transaction.device_id, trace_id=trace_id)
        span.end(output=device)
    else:
        device = {"suspicious": False, "user_count": 1, "first_seen": None}

    span = trace.span(
        name="tool.check_blacklist",
        input={
            "user_id": transaction.user_id,
            "email": transaction.email,
            "card_bin": transaction.card_bin,
        },
    )
    blacklist = await client.check_blacklist(
        transaction.user_id, transaction.email, transaction.card_bin, trace_id=trace_id
    )
    span.end(output=blacklist)

    embed_text = f"{transaction.merchant_category} {transaction.amount}"
    embedding = await embed(embed_text)
    span = trace.span(
        name="tool.search_similar_cases",
        input={"merchant_category": transaction.merchant_category, "amount": transaction.amount},
    )
    similar = await client.search_similar_cases(embedding, limit=5, trace_id=trace_id)
    span.end(output=similar)

    span = trace.span(
        name="tool.check_velocity",
        input={
            "ip_address": transaction.ip_address or "",
            "card_bin": transaction.card_bin or "",
            "window_minutes": 60,
        },
    )
    velocity = await client.check_velocity(
        ip_address=transaction.ip_address or "",
        card_bin=transaction.card_bin or "",
        window_minutes=60,
        trace_id=trace_id,
    )
    span.end(output=velocity)

    return history, ip_risk, device, blacklist, similar, velocity


def _derive_signals(
    blacklist: dict,
    ip_risk: dict,
    device: dict,
    history: dict,
    velocity: dict | None = None,
) -> Signals:
    if velocity is not None:
        velocity_flag = (
            velocity.get("distinct_users_by_ip", 0) >= 5
            or velocity.get("distinct_users_by_bin", 0) >= 5
        )
    else:
        velocity_flag = history.get("transaction_count", 0) > 20
    return Signals(
        blacklist_hit=blacklist.get("match", False),
        ip_risk_flag=ip_risk.get("risk_score", 0.0) > 0.5,
        device_fingerprint_flag=device.get("suspicious", False),
        user_history_flag=history.get("high_velocity", False),
        velocity_flag=velocity_flag,
    )


def _apply_grounding_override(verdict: Verdict) -> Verdict:
    """Enforces the grounding rule: a non-ESCALATE verdict without cited sources must escalate."""
    if verdict.cited_sources or verdict.decision == Decision.ESCALATE:
        return verdict

    return Verdict.model_validate(
        {
            **verdict.model_dump(),
            "decision": Decision.ESCALATE,
            "escalation_reason": "No grounded source could be identified to support a verdict.",
            "escalation_category": EscalationCategory.INSUFFICIENT_GROUNDING,
            "cited_sources": [],
        }
    )


async def _persist_verdict(verdict: Verdict) -> None:
    client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)
    try:
        await client.save_verdict(verdict.model_dump())
    except Exception:
        log.warn("verdict_persist_failed", transaction_id=verdict.transaction_id)
    finally:
        await client.close()


async def analyze(transaction: Transaction) -> Verdict:
    trace = start_trace(
        "analyze_transaction",
        input={
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "user_id": transaction.user_id,
        },
    )
    try:
        try:
            if IS_MOCK:
                verdict = await asyncio.wait_for(
                    _run_mock(transaction, trace),
                    timeout=settings.analyze_timeout_seconds,
                )
            else:
                verdict = await asyncio.wait_for(
                    _run_real(transaction, trace),
                    timeout=settings.analyze_timeout_seconds,
                )
        except TimeoutError:
            latency_ms = int(settings.analyze_timeout_seconds * 1000)
            msg = f"Analysis timed out after {settings.analyze_timeout_seconds}s"
            log.warn("analyze_timeout", transaction_id=transaction.transaction_id)
            trace.update(
                output={"decision": "ESCALATE", "reason": "timeout"},
                metadata={"escalated": True},
            )
            verdict = Verdict.model_validate(
                {
                    "transaction_id": transaction.transaction_id,
                    "decision": Decision.ESCALATE,
                    "risk_score": 0.5,
                    "reasoning": msg,
                    "cited_sources": [],
                    "signals": {
                        k: False
                        for k in [
                            "user_history_flag",
                            "ip_risk_flag",
                            "device_fingerprint_flag",
                            "blacklist_hit",
                            "velocity_flag",
                        ]
                    },
                    "escalation_reason": msg,
                    "latency_ms": latency_ms,
                    "model": "timeout",
                    "tool_calls": 0,
                    "langfuse_trace_id": getattr(trace, "id", ""),
                    "user_id": transaction.user_id,
                    "amount": transaction.amount,
                    "currency": transaction.currency,
                    "ip_address": transaction.ip_address,
                    "card_bin": transaction.card_bin,
                    "device_id": transaction.device_id,
                }
            )
    finally:
        langfuse.flush()

    verdict = _apply_grounding_override(verdict)
    await _persist_verdict(verdict)
    return verdict


def _summarize_tool_result(name: str, result: dict) -> str:
    """Return a one-line human-readable summary of a tool result."""
    match name:
        case "get_user_history":
            count = result.get("transaction_count", 0)
            countries = len(result.get("countries", []))
            velocity = result.get("high_velocity", False)
            return f"{count} transactions, {countries} countries, high_velocity={velocity}"
        case "get_ip_risk":
            score = result.get("risk_score", 0.0)
            vpn = result.get("is_vpn", False)
            country = result.get("country", "unknown")
            return f"score {score:.2f}, vpn={vpn}, country={country}"
        case "get_device_fingerprint":
            suspicious = result.get("suspicious", False)
            users = result.get("user_count", 1)
            return f"suspicious={suspicious}, shared_users={users}"
        case "check_blacklist":
            match = result.get("match", False)
            kind = result.get("kind") or "none"
            return f"match={match}, kind={kind}"
        case "search_similar_cases":
            cases = result.get("cases", [])
            return f"{len(cases)} similar cases retrieved"
        case "check_velocity":
            by_ip = result.get("distinct_users_by_ip", 0)
            by_bin = result.get("distinct_users_by_bin", 0)
            total = result.get("total_txns_in_window", 0)
            window = result.get("window_minutes", 60)
            return f"{by_ip} users/IP, {by_bin} users/BIN, {total} txns in {window}min"
        case _:
            return str(result)[:120]


async def stream_mock(
    transaction: Transaction,
    data_client: TesseraDataClient,
    trace_id: str,
) -> AsyncIterator[SSEEvent]:
    """Streaming mock runner — yields SSE events as each tool runs."""
    try:
        yield SSEEvent(
            type="start",
            data={"trace_id": trace_id, "mode": "mock", "model": "mock-runner-v1"},
        )

        tool_sequence = [
            ("get_user_history", {"user_id": transaction.user_id}),
            (
                "get_ip_risk",
                {"ip_address": transaction.ip_address or ""},
            ),
            (
                "get_device_fingerprint",
                {"device_id": transaction.device_id or ""},
            ),
            (
                "check_blacklist",
                {
                    "user_id": transaction.user_id,
                    "email": transaction.email or "",
                    "card_bin": transaction.card_bin or "",
                },
            ),
            (
                "search_similar_cases",
                {
                    "transaction_id": transaction.transaction_id,
                    "amount": transaction.amount,
                    "merchant_category": transaction.merchant_category or "",
                },
            ),
            (
                "check_velocity",
                {
                    "ip_address": transaction.ip_address or "",
                    "card_bin": transaction.card_bin or "",
                    "window_minutes": 60,
                },
            ),
        ]

        results: dict[str, dict] = {}
        for name, args in tool_sequence:
            yield SSEEvent(type="tool_start", data={"name": name, "args": args})
            t0 = time.monotonic()
            result = await dispatch(name, args, data_client, trace_id=trace_id)
            await asyncio.sleep(0.6)
            duration_ms = int((time.monotonic() - t0) * 1000)
            results[name] = result
            yield SSEEvent(
                type="tool_complete",
                data={
                    "name": name,
                    "duration_ms": duration_ms,
                    "summary": _summarize_tool_result(name, result),
                },
            )

        history = results.get("get_user_history", {})
        ip_risk = results.get("get_ip_risk", {})
        device = results.get("get_device_fingerprint", {})
        blacklist = results.get("check_blacklist", {})
        similar = results.get("search_similar_cases", {})
        velocity = results.get("check_velocity", None)

        signals = _derive_signals(blacklist, ip_risk, device, history, velocity)
        blacklist_hit = signals.blacklist_hit
        flag_count = sum(
            [
                signals.ip_risk_flag,
                signals.device_fingerprint_flag,
                signals.user_history_flag,
                signals.velocity_flag,
            ]
        )

        if blacklist_hit:
            decision = Decision.DECLINE
            risk_score = 0.95
            cited_sources = [
                CitedSource(
                    type=SourceType.RULE,
                    id="BLACKLIST_HIT",
                    excerpt="Entity matched fraud blacklist.",
                )
            ]
            escalation_reason = None
            escalation_category = None
        elif flag_count >= 2:
            decision = Decision.ESCALATE
            risk_score = 0.65
            cited_sources = [
                CitedSource(
                    type=SourceType.RULE,
                    id="MULTI_SIGNAL_RISK",
                    excerpt="Multiple risk signals present; escalating for human review.",
                )
            ]
            escalation_reason = "Multiple risk signals detected; human review required."
            escalation_category = EscalationCategory.LOW_CONFIDENCE
        else:
            decision = Decision.APPROVE
            risk_score = 0.10
            cited_sources = [
                CitedSource(
                    type=SourceType.RULE,
                    id="NO_RISK_FLAGS",
                    excerpt="No risk signals detected across all checks.",
                )
            ]
            escalation_reason = None
            escalation_category = None

        cases = similar.get("cases", [])
        if cases:
            top = cases[0]
            cited_sources.append(
                CitedSource(
                    type=SourceType.CASE,
                    id=top.get("id", ""),
                    excerpt=(top.get("reasoning", "") or "")[:200],
                )
            )

        reasoning = (
            f"User history: {history.get('transaction_count', 0)} transactions, "
            f"high_velocity={history.get('high_velocity', False)}. "
            f"IP risk score: {ip_risk.get('risk_score', 0.0)}, "
            f"is_vpn={ip_risk.get('is_vpn', False)}. "
            f"Device suspicious={device.get('suspicious', False)}, "
            f"shared_users={device.get('user_count', 1)}. "
            f"Blacklist match={blacklist.get('match', False)}."
        )

        verdict_dict = {
            "transaction_id": transaction.transaction_id,
            "decision": decision,
            "risk_score": risk_score,
            "reasoning": reasoning,
            "cited_sources": [s.model_dump() for s in cited_sources],
            "signals": signals.model_dump(),
            "escalation_reason": escalation_reason,
            "escalation_category": escalation_category,
            "latency_ms": 0,
            "model": "mock-runner-v1",
            "tool_calls": 6,
            "langfuse_trace_id": trace_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "ip_address": transaction.ip_address,
            "card_bin": transaction.card_bin,
            "device_id": transaction.device_id,
        }

        verdict = _apply_grounding_override(Verdict.model_validate(verdict_dict))
        await _persist_verdict(verdict)

        yield SSEEvent(type="verdict", data=verdict.model_dump())
        yield SSEEvent(type="done", data={})

    except Exception as exc:
        log.warn("stream_mock_error", error=str(exc), transaction_id=transaction.transaction_id)
        yield SSEEvent(type="error", data={"message": str(exc)})
        yield SSEEvent(type="done", data={})


async def stream_real(
    transaction: Transaction,
    data_client: TesseraDataClient,
    trace_id: str,
) -> AsyncIterator[SSEEvent]:
    """Streaming real runner — yields SSE events as Claude calls each tool."""
    try:
        stream_start = time.monotonic()
        yield SSEEvent(
            type="start",
            data={"trace_id": trace_id, "mode": "real", "model": "claude-sonnet-4-6"},
        )

        ai_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        messages: list[dict] = [
            {
                "role": "user",
                "content": f"Analyze this transaction:\n{transaction.model_dump_json(indent=2)}",
            }
        ]

        tool_call_count = 0
        final_text: str | None = None
        total_in = 0
        total_out = 0

        try:
            while True:
                for attempt in range(settings.max_analyze_retries + 1):
                    try:
                        response = await ai_client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=2048,
                            system=_SYSTEM_PROMPT,
                            tools=TOOLS,  # type: ignore[arg-type]
                            messages=messages,  # type: ignore[arg-type]
                        )
                        break
                    except APIStatusError as e:
                        if e.status_code in {429, 529} and attempt < settings.max_analyze_retries:
                            wait = 2**attempt
                            log.warn(
                                "anthropic_retryable_error",
                                status=e.status_code,
                                attempt=attempt,
                                wait_s=wait,
                                transaction_id=transaction.transaction_id,
                            )
                            await asyncio.sleep(wait)
                            continue
                        raise

                total_in += response.usage.input_tokens
                total_out += response.usage.output_tokens

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text = block.text
                            break
                    break

                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            if tool_call_count >= 10:
                                log.warn(
                                    "max_tool_calls_exceeded",
                                    transaction_id=transaction.transaction_id,
                                )
                                break

                            yield SSEEvent(
                                type="tool_start",
                                data={"name": block.name, "args": block.input},
                            )
                            t0 = time.monotonic()
                            result = await dispatch(
                                block.name,
                                block.input,
                                data_client,
                                trace_id=trace_id,
                            )
                            duration_ms = int((time.monotonic() - t0) * 1000)
                            tool_call_count += 1
                            yield SSEEvent(
                                type="tool_complete",
                                data={
                                    "name": block.name,
                                    "duration_ms": duration_ms,
                                    "summary": _summarize_tool_result(block.name, result),
                                },
                            )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps(result),
                                }
                            )

                    if tool_call_count >= 10:
                        break

                    messages.append({"role": "assistant", "content": response.content})  # type: ignore[arg-type]
                    messages.append({"role": "user", "content": tool_results})
                    continue

                break

        finally:
            langfuse.flush()

        latency_ms = int((time.monotonic() - stream_start) * 1000)
        _max_calls_msg = "Agent reached maximum tool call limit without producing a verdict."

        if tool_call_count >= 10 or final_text is None:
            escalate_verdict = Verdict.model_validate(
                {
                    "transaction_id": transaction.transaction_id,
                    "decision": Decision.ESCALATE,
                    "risk_score": 0.5,
                    "reasoning": _max_calls_msg,
                    "cited_sources": [],
                    "signals": {
                        k: False
                        for k in [
                            "user_history_flag",
                            "ip_risk_flag",
                            "device_fingerprint_flag",
                            "blacklist_hit",
                            "velocity_flag",
                        ]
                    },
                    "escalation_reason": _max_calls_msg,
                    "latency_ms": latency_ms,
                    "model": "claude-sonnet-4-6",
                    "tool_calls": tool_call_count,
                    "langfuse_trace_id": trace_id,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
                    "user_id": transaction.user_id,
                    "amount": transaction.amount,
                    "currency": transaction.currency,
                    "ip_address": transaction.ip_address,
                    "card_bin": transaction.card_bin,
                    "device_id": transaction.device_id,
                }
            )
            await _persist_verdict(escalate_verdict)
            yield SSEEvent(type="verdict", data=escalate_verdict.model_dump())
            yield SSEEvent(type="done", data={})
            return

        try:
            raw = _extract_json(final_text)
        except (json.JSONDecodeError, ValueError):
            parse_fail = Verdict.model_validate(
                {
                    "transaction_id": transaction.transaction_id,
                    "decision": Decision.ESCALATE,
                    "risk_score": 0.5,
                    "reasoning": final_text,
                    "cited_sources": [],
                    "signals": {
                        k: False
                        for k in [
                            "user_history_flag",
                            "ip_risk_flag",
                            "device_fingerprint_flag",
                            "blacklist_hit",
                            "velocity_flag",
                        ]
                    },
                    "escalation_reason": "Agent produced non-parseable output.",
                    "latency_ms": latency_ms,
                    "model": "claude-sonnet-4-6",
                    "tool_calls": tool_call_count,
                    "langfuse_trace_id": trace_id,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
                    "user_id": transaction.user_id,
                    "amount": transaction.amount,
                    "currency": transaction.currency,
                    "ip_address": transaction.ip_address,
                    "card_bin": transaction.card_bin,
                    "device_id": transaction.device_id,
                }
            )
            await _persist_verdict(parse_fail)
            yield SSEEvent(type="verdict", data=parse_fail.model_dump())
            yield SSEEvent(type="done", data={})
            return

        cited_sources_raw = raw.get("cited_sources", [])
        cited_sources = [CitedSource.model_validate(s) for s in cited_sources_raw]
        signals_raw = raw.get(
            "signals",
            {
                "user_history_flag": False,
                "ip_risk_flag": False,
                "device_fingerprint_flag": False,
                "blacklist_hit": False,
                "velocity_flag": False,
            },
        )
        verdict = Verdict.model_validate(
            {
                "transaction_id": transaction.transaction_id,
                "decision": raw.get("decision", Decision.ESCALATE),
                "risk_score": float(raw.get("risk_score", 0.5)),
                "reasoning": raw.get("reasoning", ""),
                "cited_sources": [s.model_dump() for s in cited_sources],
                "signals": signals_raw,
                "escalation_reason": raw.get("escalation_reason"),
                "escalation_category": raw.get("escalation_category"),
                "latency_ms": latency_ms,
                "model": "claude-sonnet-4-6",
                "tool_calls": tool_call_count,
                "langfuse_trace_id": trace_id,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cost_usd": compute_cost_usd(total_in, total_out, "claude-sonnet-4-6"),
                "user_id": transaction.user_id,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "ip_address": transaction.ip_address,
                "card_bin": transaction.card_bin,
                "device_id": transaction.device_id,
            }
        )
        verdict = _apply_grounding_override(verdict)
        await _persist_verdict(verdict)
        yield SSEEvent(type="verdict", data=verdict.model_dump())
        yield SSEEvent(type="done", data={})

    except Exception as exc:
        log.warn("stream_real_error", error=str(exc), transaction_id=transaction.transaction_id)
        yield SSEEvent(type="error", data={"message": str(exc)})
        yield SSEEvent(type="done", data={})
