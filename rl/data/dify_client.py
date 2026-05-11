"""Minimal Dify-compatible signed client for collection scripts."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class DifyConfig:
    base_url: str
    access_key_id: str
    access_key_secret: str
    agent_id: str
    user: str


def mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}****{value[-4:]}"


def from_env(base_url: str, user: str) -> DifyConfig:
    access_key_id = os.getenv("DIFY_ACCESS_KEY_ID", "").strip()
    access_key_secret = os.getenv("DIFY_ACCESS_KEY_SECRET", "").strip()
    agent_id = os.getenv("DIFY_ALFWORLD_AGENT_ID", "").strip()
    if not access_key_id or not access_key_secret or not agent_id:
        raise RuntimeError("Missing Dify env vars: DIFY_ACCESS_KEY_ID / DIFY_ACCESS_KEY_SECRET / DIFY_ALFWORLD_AGENT_ID")
    return DifyConfig(
        base_url=base_url.rstrip("/"),
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        agent_id=agent_id,
        user=user,
    )


def _sign_params(method: str, url: str, access_key_id: str, access_key_secret: str) -> Dict[str, str]:
    target = url.lstrip("https://").lstrip("http://")
    sign_params = {
        "AccessKeyId": access_key_id,
        "Expires": 60,
        "Timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    sorted_str = urllib.parse.urlencode({k: sign_params[k] for k in sorted(sign_params.keys())})
    sign_str = f"{method.upper()}{target}?{sorted_str}"
    digest = hmac.new(access_key_secret.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha1).digest()
    signature = urllib.parse.quote(base64.b64encode(digest), safe="")
    return {
        "AccessKeyId": sign_params["AccessKeyId"],
        "Expires": str(sign_params["Expires"]),
        "Timestamp": sign_params["Timestamp"],
        "Signature": signature,
    }


class DifyClient:
    def __init__(self, config: DifyConfig, timeout_s: float = 60.0):
        self.config = config
        self.timeout_s = timeout_s

    def _signed_url(self, api_path: str, method: str = "POST") -> str:
        url = f"{self.config.base_url}{api_path}"
        params = _sign_params(method=method, url=url, access_key_id=self.config.access_key_id, access_key_secret=self.config.access_key_secret)
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{url}?{query}"

    def create_conversation(self, inputs: Optional[Dict[str, Any]] = None) -> str:
        url = self._signed_url("/agent/v1/create-conversation")
        body = {
            "agent_id": self.config.agent_id,
            "user": self.config.user,
            "inputs": inputs or {},
            "created_at": int(dt.datetime.now().timestamp() * 1000),
        }
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=self.timeout_s)
        payload = resp.json()
        if "error" in payload or not payload.get("conversation_id"):
            raise RuntimeError(f"create_conversation failed: {payload}")
        return str(payload["conversation_id"])

    def send_message(self, conversation_id: str, message: str) -> str:
        url = self._signed_url("/agent/v1/chat-messages")
        body = {
            "agent_id": self.config.agent_id,
            "conversation_id": conversation_id,
            "user": self.config.user,
            "query": [
                {
                    "content": message,
                    "content_type": "text",
                    "created_at": int(dt.datetime.now().timestamp() * 1000),
                }
            ],
            "inputs": {},
            "response_mode": "blocking",
        }
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=self.timeout_s)
        payload = resp.json()
        if "error" in payload or "code" in payload:
            raise RuntimeError(f"send_message failed: {payload}")
        answer = payload.get("answer") or []
        if not answer or not isinstance(answer, list):
            raise RuntimeError(f"empty answer payload: {payload}")
        content = answer[0].get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"invalid answer content: {payload}")
        return content

