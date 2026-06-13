import httpx
import structlog

log = structlog.get_logger(__name__)

_EMPTY_USER_HISTORY: dict = {
    "transaction_count": 0,
    "avg_amount": 0.0,
    "countries": [],
    "last_txn_at": None,
    "high_velocity": False,
}
_EMPTY_IP_RISK: dict = {"risk_score": 0.0, "is_vpn": False, "country": "unknown"}
_EMPTY_DEVICE_FINGERPRINT: dict = {"suspicious": False, "user_count": 1, "first_seen": None}
_EMPTY_BLACKLIST: dict = {"match": False, "kind": None, "reason": None}


class TesseraDataClient:
    def __init__(self, base_url: str, internal_key: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"X-Internal-Key": internal_key},
            timeout=3.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _trace_headers(self, trace_id: str | None) -> dict[str, str]:
        if trace_id:
            return {"X-Trace-ID": trace_id}
        return {}

    async def get_user_history(self, user_id: str, trace_id: str | None = None) -> dict:
        try:
            r = await self._client.get(
                f"/users/{user_id}/history", headers=self._trace_headers(trace_id)
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="get_user_history",
                user_id=user_id,
                error=str(exc),
            )
            return _EMPTY_USER_HISTORY

    async def get_ip_risk(self, ip_address: str, trace_id: str | None = None) -> dict:
        try:
            r = await self._client.get(
                f"/ip/{ip_address}/risk", headers=self._trace_headers(trace_id)
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="get_ip_risk",
                ip_address=ip_address,
                error=str(exc),
            )
            return _EMPTY_IP_RISK

    async def get_device_fingerprint(self, device_id: str, trace_id: str | None = None) -> dict:
        try:
            r = await self._client.get(
                f"/devices/{device_id}/fingerprint", headers=self._trace_headers(trace_id)
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="get_device_fingerprint",
                device_id=device_id,
                error=str(exc),
            )
            return _EMPTY_DEVICE_FINGERPRINT

    async def check_blacklist(
        self,
        user_id: str,
        email: str,
        card_bin: str,
        trace_id: str | None = None,
    ) -> dict:
        try:
            r = await self._client.get(
                "/blacklist/check",
                params={"user_id": user_id, "email": email, "card_bin": card_bin},
                headers=self._trace_headers(trace_id),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="check_blacklist",
                user_id=user_id,
                error=str(exc),
            )
            return _EMPTY_BLACKLIST

    async def search_similar_cases(
        self,
        embedding: list[float],
        limit: int = 5,
        trace_id: str | None = None,
    ) -> dict:
        try:
            r = await self._client.post(
                "/cases/similar",
                json={"embedding": embedding, "limit": limit},
                headers=self._trace_headers(trace_id),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="search_similar_cases",
                error=str(exc),
            )
            return {"cases": []}

    async def save_case(self, case_data: dict, trace_id: str | None = None) -> dict:
        try:
            r = await self._client.post(
                "/cases",
                json=case_data,
                headers=self._trace_headers(trace_id),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="save_case",
                transaction_id=case_data.get("transaction_id"),
                error=str(exc),
            )
            return {"id": "", "transaction_id": case_data.get("transaction_id", "")}

    async def save_verdict(self, verdict_data: dict, trace_id: str = "") -> dict:
        """POST /verdicts — persist a verdict. Returns {"id": ..., "transaction_id": ...}"""
        try:
            r = await self._client.post(
                "/verdicts",
                json=verdict_data,
                headers=self._trace_headers(trace_id),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="save_verdict",
                transaction_id=verdict_data.get("transaction_id"),
                error=str(exc),
            )
            return {"id": "", "transaction_id": verdict_data.get("transaction_id", "")}

    async def list_verdicts(self, limit: int = 50, offset: int = 0, trace_id: str = "") -> dict:
        """GET /verdicts?limit=N&offset=N — returns {"verdicts": [...], "total": N}"""
        try:
            r = await self._client.get(
                "/verdicts",
                params={"limit": limit, "offset": offset},
                headers=self._trace_headers(trace_id),
            )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warn(
                "tessera_data_error",
                method="list_verdicts",
                error=str(exc),
            )
            return {"verdicts": [], "total": 0}
