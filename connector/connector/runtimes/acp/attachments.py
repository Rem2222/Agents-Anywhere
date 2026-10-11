from __future__ import annotations

import base64
import re
from typing import Any

from connector.runtime_protocol import RuntimeAttachment, RuntimeInvalidRequestError
from connector.runtime_protocol.host import RuntimeHostClient


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


async def image_blocks(
    host: RuntimeHostClient,
    session_id: str,
    attachments: tuple[RuntimeAttachment, ...],
) -> list[dict[str, Any]]:
    """Download image attachments into ACP ``image`` content blocks."""
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attachment in attachments:
        if not re.fullmatch(r"file_[\w-]{1,128}", attachment.file_id) or attachment.file_id in seen:
            raise RuntimeInvalidRequestError("Invalid or duplicate ACP attachment")
        seen.add(attachment.file_id)
        downloaded = await host.attachment_download(session_id, attachment.file_id)
        media_type = (downloaded.media_type or "").split(";", 1)[0].strip().lower()
        if not media_type.startswith("image/"):
            raise RuntimeInvalidRequestError("ACP attachments must be images")
        blocks.append(
            {
                "type": "image",
                "mimeType": media_type or "application/octet-stream",
                "data": base64.b64encode(downloaded.content).decode("ascii"),
            }
        )
    return blocks
