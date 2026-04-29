import json
import hashlib
import logging
from datetime import datetime, timezone
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

ENTITY_EXTRACT_SYSTEM = """You are a procurement data analyst. Extract all named entities from the document chunk below.

Entity types:
- vendor: supplier company or individual
- customer: buying department or company
- po: Purchase Order number (pattern: PO-YYYY-NNN)
- invoice: Invoice number (pattern: INV-YYYY-NNN)
- grn: Goods Receipt Note number (pattern: GRN-YYYY-NNN)
- project: project or cost centre name
- item: specific product or service name
- contact: person name with role

For each entity include any document references it links to (e.g. an Invoice entity should list its PO reference).
"""

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "store_entities",
        "description": "Store all entities extracted from the document chunk.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string"},
                            "entity_type": {
                                "type": "string",
                                "enum": ["vendor","customer","po","invoice","grn","project","item","contact","unknown"]
                            },
                            "entity_value": {"type": "string", "description": "Uppercase canonical form"},
                            "entity_aliases": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "related_entity_refs": {
                                "type": "array",
                                "description": "Document IDs or entity names this entity is linked to",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["entity_name", "entity_type", "entity_value"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}


def _make_entity_id(entity_type: str, entity_value: str) -> str:
    raw = f"{entity_type}:{entity_value.upper()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class EntityExtractor:
    def __init__(self, openai_client: AsyncOpenAI, chat_deployment: str):
        self.client = openai_client
        self.deployment = chat_deployment

    async def extract_entities_from_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        source_file: str,
        allowed_users: list[str],
        allowed_groups: list[str],
    ) -> list[dict]:
        if len(chunk_text.split()) < 30:
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": ENTITY_EXTRACT_SYSTEM},
                    {"role": "user", "content": f"Document chunk:\n\n{chunk_text}"},
                ],
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "store_entities"}},
                temperature=0,
                max_tokens=2000,
            )
        except Exception as e:
            logger.warning("Entity extraction LLM call failed for chunk %s: %s", chunk_id, e)
            return []

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            return []

        try:
            args = json.loads(tool_calls[0].function.arguments)
            raw_entities = args.get("entities", [])
        except (json.JSONDecodeError, KeyError):
            return []

        now = datetime.now(timezone.utc).isoformat()
        result = []
        for e in raw_entities:
            entity_type = e.get("entity_type", "unknown")
            entity_value = e.get("entity_value", e.get("entity_name", "")).upper()
            entity_id = _make_entity_id(entity_type, entity_value)

            related = []
            for ref in e.get("related_entity_refs", []):
                ref_type = _infer_ref_type(ref)
                related.append({
                    "entity_id": _make_entity_id(ref_type, ref.upper()),
                    "relationship_type": _infer_relationship(entity_type, ref_type),
                    "strength": 1.0,
                })

            result.append({
                "entity_id": entity_id,
                "entity_name": e.get("entity_name", entity_value),
                "entity_type": entity_type,
                "entity_value": entity_value,
                "entity_aliases": e.get("entity_aliases", []),
                "related_entities": json.dumps(related),
                "source_chunks": [chunk_id],
                "source_files": [source_file],
                "allowedUsers": allowed_users,
                "allowedGroups": allowed_groups,
                "mention_count": 1,
                "last_seen": now,
            })

        return result


def _infer_ref_type(ref: str) -> str:
    r = ref.upper()
    if r.startswith("PO-"):
        return "po"
    if r.startswith("INV-"):
        return "invoice"
    if r.startswith("GRN-"):
        return "grn"
    return "unknown"


def _infer_relationship(from_type: str, to_type: str) -> str:
    mapping = {
        ("invoice", "po"): "fulfills",
        ("grn", "po"): "closes",
        ("grn", "invoice"): "validates",
        ("po", "vendor"): "issued_to",
        ("invoice", "vendor"): "from",
        ("grn", "vendor"): "received_from",
        ("po", "project"): "for_project",
        ("invoice", "project"): "charged_to",
    }
    return mapping.get((from_type, to_type), "related_to")
