"""
Product Intelligence and Customer Service Platform — ShopMart Retail
======================================================================
Build this file in 6 phases, one day's skills at a time. It is the Retail
counterpart to loan_origination_assistant.py (Finance) — pick ONE of the two
to implement; both are scaffolded the same way.

Phase 1  (Day 1)   — Secure client · enrichment system prompt · Q&A system prompt
Phase 2  (Day 2)   — ProductRecord schema · parse() with retry · enrichment pipeline
Phase 3  (Day 2)   — ProductConversationManager · multi-turn Q&A session
Phase 4  (Day 3)   — Tool definitions (inventory/price/vendor spec) · manual agentic loop
Phase 5  (Day 3-4) — Chroma-backed RAG over catalogue + vendor specs · Voyage embeddings
Phase 6  (Day 4)   — Enrichment accuracy + Q&A faithfulness evaluation

Run:
    python product_platform.py
"""

# ── Imports (provided) ─────────────────────────────────────────────────────────
import json
import os
import re
import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

import anthropic
import logging

# ── Constants (provided) ───────────────────────────────────────────────────────
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
TOKEN_WARN_THRESHOLD = 30_000       # print a warning once the Q&A session crosses this
TOKEN_COMPACT_THRESHOLD = 60_000    # summarise-and-reset once the Q&A session crosses this
MAX_PARSE_RETRIES = 2
TOP_K_CHUNKS = 3
RAW_CATALOGUE_PATH = Path("data/retail_products.txt")
VENDOR_SPEC_DIR = Path("data/vendor_specs")
EVAL_LOG_PATH = Path("eval_logs/product_platform_v1.jsonl")
CHROMA_COLLECTION_NAME = "shopmart_catalogue"
FALLBACK_RESPONSE = (
    "I don't have that specification on file — I recommend checking with the "
    "vendor or contacting our support team."
)
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



# ── Mock data (provided — do not modify) ──────────────────────────────────────
# Simulates the three real-time systems the Q&A assistant calls via tool use.

INVENTORY_DB = {
    "SKU-E001": {"available": True,  "quantity": 12, "warehouse": "Whitefield WH"},
    "SKU-A002": {"available": False, "quantity": 0,  "warehouse": "N/A"},
    "SKU-H003": {"available": True,  "quantity": 47, "warehouse": "Hosur Plant"},
}

PRICE_DB = {
    "SKU-E001": {"price_inr": 124_990.0, "discount_pct": 8, "offer_ends": "2026-07-26"},
    "SKU-A002": {"price_inr": 3_495.0,   "discount_pct": 0, "offer_ends": "N/A"},
    "SKU-H003": {"price_inr": 2_199.0,   "discount_pct": 0, "offer_ends": "N/A"},
}

# Keyed by (sku, normalised spec_field) — mirrors what a vendor spec-sheet
# lookup service would return for a single named field.
VENDOR_SPEC_DB = {
    ("SKU-E001", "ram"):               {"field": "RAM", "value": "32GB LPDDR5, soldered — not user-upgradeable", "source": "vendor_sheet"},
    ("SKU-E001", "thunderbolt"):       {"field": "Ports", "value": "2x Thunderbolt 4 (USB-C), 1x USB-C 3.2 Gen 2", "source": "vendor_sheet"},
    ("SKU-E001", "ports"):             {"field": "Ports", "value": "2x Thunderbolt 4 (USB-C), 1x USB-C 3.2 Gen 2", "source": "vendor_sheet"},
    ("SKU-E001", "warranty"):          {"field": "Warranty", "value": "12 months international warranty, India-serviceable via Dell ExpressService", "source": "vendor_sheet"},
    ("SKU-A002", "water_resistance"):  {"field": "Water Resistance", "value": "30 metres (3 ATM) — splash resistant only, not for swimming", "source": "vendor_sheet"},
    ("SKU-A002", "strap_material"):    {"field": "Strap Material", "value": "Genuine leather, brown", "source": "vendor_sheet"},
    ("SKU-H003", "isi_certification"): {"field": "ISI Certification", "value": "IS 2347:2017 certified", "source": "vendor_sheet"},
    ("SKU-H003", "warranty"):          {"field": "Warranty", "value": "24 months on the cooker body, 12 months on gasket/safety valve", "source": "vendor_sheet"},
}

# 6-turn test conversation about the Dell XPS 15 (SKU-E001). The final turn
# asks about a spec that is NOT in the vendor sheet, to exercise the fallback.
TEST_CONVERSATION = [
    "Hi, I'm looking at the Dell XPS 15, SKU-E001. Can I upgrade the RAM myself later?",
    "Does it support Thunderbolt?",
    "What's the warranty like in India?",
    "Is it in stock right now?",
    "What's the current price, with any active discount?",
    "One last thing — does it come in a silver colour option?",
]

# Ground-truth ProductRecord field values for 5 products (Phase 6 enrichment eval)
ENRICHMENT_GOLDEN_SET = [
    {"sku": "SKU-E001", "ground_truth": {
        "brand": "Dell", "category": "electronics", "price_inr": 124_990.0, "in_stock": True}},
    {"sku": "SKU-E002", "ground_truth": {
        "brand": "boAt", "category": "electronics", "price_inr": 1_499.0, "in_stock": True}},
    {"sku": "SKU-A002", "ground_truth": {
        "brand": "Titan", "category": "apparel", "price_inr": 3_495.0, "in_stock": False}},
    {"sku": "SKU-H003", "ground_truth": {
        "brand": "Prestige", "category": "homeware", "price_inr": 2_199.0, "in_stock": True}},
    {"sku": "SKU-B001", "ground_truth": {
        "brand": "Himalaya Herbals", "category": "beauty", "price_inr": 165.0, "in_stock": True}},
]

# 5 customer queries with the expected grounding source (Phase 6 Q&A eval)
QA_GOLDEN_SET = [
    {"query": "What is the RAM capacity of the Dell XPS 15?", "sku": "SKU-E001", "expected_source": "catalogue"},
    {"query": "Does the Dell XPS 15 support Thunderbolt?", "sku": "SKU-E001", "expected_source": "vendor_spec"},
    {"query": "Is the Titan Kairos watch safe to wear while swimming?", "sku": "SKU-A002", "expected_source": "vendor_spec"},
    {"query": "Is the Prestige Svachh cooker ISI certified?", "sku": "SKU-H003", "expected_source": "vendor_spec"},
    {"query": "Does the Dell XPS 15 come in a silver colour option?", "sku": "SKU-E001", "expected_source": "fallback"},
]


# ── Raw catalogue loader (provided) ────────────────────────────────────────────

def load_raw_products(path: Path = RAW_CATALOGUE_PATH) -> dict[str, str]:
    """Parse data/retail_products.txt into {sku: raw_description_text}.

    The file uses '### SKU-XXX' headers to delimit each vendor's raw
    submission. This is plain file parsing (not a taught skill) — provided
    so Phase 2 can focus on the extraction prompt and parse/retry loop.
    """
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^### (SKU-[A-Z0-9-]+)\s*$", text, flags=re.MULTILINE)[1:]
    return {
        sku: desc.strip()
        for sku, desc in zip(blocks[0::2], blocks[1::2])
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Secure Foundation (Day 1 skills)
# ═══════════════════════════════════════════════════════════════════════════════

def make_client() -> anthropic.Anthropic:
    """Initialise the shared Anthropic client used by both the enrichment
    pipeline (batch) and the Q&A assistant (real-time).

    TODO:
    - Call load_dotenv() to pick up .env
    - Read ANTHROPIC_API_KEY with os.environ.get()
    - Raise EnvironmentError with a descriptive message if it is absent
    - Return anthropic.Anthropic() — no api_key= argument; SDK reads env automatically
    """

    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy shared/.env.example to .env and add your key."
        )
    return anthropic.Anthropic()
    # raise NotImplementedError("Phase 1 ▸ implement make_client()")


# Write the enrichment prompt here. Applied per-product in run_enrichment_pipeline().
ENRICHMENT_SYSTEM = """

You are the ShopMart product data enrichment assistant.
Your role: extract and enrich structured product catalogue fields from the provided product text accurately and consistently.

CONSTRAINTS:

- Extract only information that is explicitly stated or can be reasonably inferred from the provided text.
- If a product category or subcategory is not explicitly stated, infer the most appropriate values from the product name, description, specifications, or other available context.
- Normalize all prices to a plain INR float (without currency symbols or formatting). Accept formats including:
    - "₹1,299" → 1299.0
    - "Rs 1299" → 1299.0
    - "INR 1,299.00" → 1299.0
- Prices written in words (e.g. "one lakh ten thousand rupees") → 110000.0
- Do not fabricate information. If a field cannot be determined or reasonably inferred from the source text, return null for that field.
- Never invent specifications, dimensions, materials, features, ratings, certifications, or any other attribute that is not supported by the source.
- Preserve factual accuracy over completeness.

OUTPUT CONTRACT:
- Output only valid JSON.
- The JSON must conform exactly to the ProductRecord schema.
- Do not include markdown, code fences, explanations, comments, or additional text.
- Every field must contain either:
    - the extracted or inferred value (where permitted), or
    - null if the value cannot be determined.

"""

"""
TODO (Phase 1): Write the enrichment system prompt.

Required elements:
1. Role definition   — ShopMart product data specialist extracting catalogue fields
2. Inference rule    — infer category/subcategory from context when not stated
3. Price rule        — normalise "₹1,299", "Rs 1299", "INR 1,299.00", and prices
                        written in words (e.g. "one lakh ten thousand rupees")
                        to a plain INR float
4. No fabrication    — return null for fields that genuinely cannot be inferred;
                        never invent a specification that isn't in the source text
5. Output contract   — output only valid JSON matching the ProductRecord schema
"""

# Write the Q&A system prompt here. {product_context} is filled in per-turn in
# Phase 4/5 with retrieved catalogue + vendor-spec chunks.
QA_SYSTEM = """
You are ShopMart's knowledgeable product advisor.
Your role: help customers understand products and make informed purchasing decisions using the product information available to you.

CONSTRAINTS:
- Answer ONLY from the retrieved product catalogue data and vendor specifications provided in the conversation.
- Use the following product context as your sole source of truth:
    "{product_context}"
- If the requested specification or information is not available in the provided context, respond with exactly:
    "{fallback}"
- Never use outside knowledge, make assumptions, or speculate.
- Never invent warranty periods, compatibility claims, or certification statuses.
- If multiple products are present in the context, ensure your answer refers to the correct product.

FORMAT:
- Respond in a warm, helpful, and customer-friendly tone.
- Keep answers concise and easy to understand.
- When appropriate, summarize key product features before answering the specific question.
- Do not mention internal systems, retrieval processes, or missing context.
- Do not include information that is not supported by the provided catalogue data or vendor specifications.
"""


"""
TODO (Phase 1): Write the Q&A assistant system prompt.

Required elements:
1. Role definition    — ShopMart's knowledgeable product advisor
2. Grounding rule     — answer only from retrieved catalogue data and vendor specs
3. Tone               — warm, helpful, appropriate for retail customers
4. Fallback           — when a spec is not available, respond with exactly:
                         "{fallback}"
5. Hard constraint    — never invent warranty periods, compatibility claims,
                         or certification statuses

Relevant product context:
{{product_context}}
""".format(fallback=FALLBACK_RESPONSE)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Structured Catalogue Enrichment (Day 2 skills)
# ═══════════════════════════════════════════════════════════════════════════════

class ProductRecord(BaseModel):
    """Validated catalogue record produced by the enrichment pipeline.

    TODO (Phase 2):
    - Replace each `Any` placeholder below with the correct type — `Any` is
      just a Pydantic-safe stand-in so this class can be imported before
      Phase 2 is implemented
    - Add a @field_validator for price_inr checking it is > 0
    - Remember: in Pydantic v2, @classmethod must appear ABOVE @field_validator
    """

    sku: str
    name: str
    brand: Optional[str]
    category: Literal["electronics","apparel","homeware","beauty","grocery","sports","other"]
    subcategory: str
    price_inr: float                           # must be > 0
    mrp_inr: Optional[float]                   # original price if discounted
    key_features: list[str]                    # 3–6 bullet points
    specifications: dict[str, str]             # e.g. {"RAM": "16GB", "Storage": "512GB SSD"}
    in_stock: bool
    warranty_months: Optional[int]
    care_instructions: Optional[str]           # relevant for apparel/homeware

    @classmethod
    @field_validator("price_inr") 
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price_inr must be positive")
        return v


# def extract_product_record(
#     client: anthropic.Anthropic,
#     sku: str,
#     raw_description: str,
# ) -> ProductRecord:
#     """Extract and validate a ProductRecord from one raw vendor description.

#     Uses client.messages.parse() and retries on ValidationError.

#     TODO (Phase 2):
#     - Build a messages list: a single user turn containing the sku and
#       raw_description, asking Claude to extract all ProductRecord fields
#     - Call client.messages.parse(model, max_tokens, system=ENRICHMENT_SYSTEM,
#       messages=messages, output_format=ProductRecord)
#     - Return response.parsed_output on success
#     - On ValidationError: append the assistant response and error details, then retry
#     - After MAX_PARSE_RETRIES attempts, re-raise the last ValidationError
#     """
#     messages = [{
#         "role": "user",
#         "content": (
#             f"Extract a ProductRecord from this raw vendor submission:\n\n"
#             f"SKU: {sku}\n"
#             f"Description: {raw_description}\n\n"
#             f"Return ONLY valid JSON conforming to the ProductRecord schema. "
#             f"Do not include markdown fences, explanations, or comments."
#         )
#     }]

#     for attempt in range(MAX_PARSE_RETRIES + 1):
#         try:
#             response = client.messages.parse(
#                 model=MODEL,
#                 max_tokens=512,
#                 system=ENRICHMENT_SYSTEM,
#                 messages=messages,
#                 output_format=ProductRecord,
#             )
#             return response.parsed_output
#         except ValidationError as e:
#             if attempt == MAX_PARSE_RETRIES:
#                 raise RuntimeError(
#                     f"Failed to extract ProductRecord for {sku} after {MAX_PARSE_RETRIES} retries. "
#                     f"Last error: {e}"
#                 ) from e
#             # client.messages.parse() raises ValidationError from within the call,
#             # so there's no response text to echo back. The error message names
#             # the offending field and value, which is enough for Claude to self-correct.
#             messages.append({
#                 "role": "user",
#                 "content": (
#                     f"Your previous output failed schema validation:\n{e}\n\n"
#                     f"Please return a corrected JSON object that matches the ProductRecord schema exactly."
#                 ),
#             })

def extract_product_record(
    client: anthropic.Anthropic,
    sku: str,
    raw_description: str,
) -> ProductRecord:
    """Extract and validate a ProductRecord from one raw vendor description.

    Uses client.messages.parse() and retries on ValidationError.

    TODO (Phase 2):
    - Build a messages list: a single user turn containing the sku and
      raw_description, asking Claude to extract all ProductRecord fields
    - Call client.messages.parse(model, max_tokens, system=ENRICHMENT_SYSTEM,
      messages=messages, output_format=ProductRecord)
    - Return response.parsed_output on success
    - On ValidationError: append the assistant response and error details, then retry
    - After MAX_PARSE_RETRIES attempts, re-raise the last ValidationError
    """
    messages = [{
        "role": "user",
        "content": (
            f"Extract a ProductRecord from this raw vendor submission:\n\n"
            f"SKU: {sku}\n"
            f"Description: {raw_description}\n\n"
            f"Return ONLY valid JSON conforming to the ProductRecord schema. "
            f"Do not include markdown fences, explanations, or comments."
        )
    }]

    for attempt in range(MAX_PARSE_RETRIES + 1):
        try:
            logging.debug(f"Attempt {attempt + 1} to parse SKU: {sku}")
            logging.debug(f"Messages: {messages}")
            response = client.messages.parse(
                model=MODEL,
                max_tokens=512,
                temperature=0,
                system=ENRICHMENT_SYSTEM,
                messages=messages,
                output_format=ProductRecord,
            )
            logging.debug(f"Response: {response}")
            return response.parsed_output
        except ValidationError as e:
            logging.error(f"Validation error on attempt {attempt + 1} for SKU {sku}: {e}")
            if attempt == MAX_PARSE_RETRIES:
                raise RuntimeError(
                    f"Failed to extract ProductRecord for {sku} after {MAX_PARSE_RETRIES} retries. "
                    f"Last error: {e}"
                ) from e
            # client.messages.parse() raises ValidationError from within the call,
            # so there's no response text to echo back. The error message names
            # the offending field and value, which is enough for Claude to self-correct.
            messages.append({
                "role": "user",
                "content": (
                    f"Your previous output failed schema validation:\n{e}\n\n"
                    f"Please return a corrected JSON object that matches the ProductRecord schema exactly."
                ),
            })
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1} for SKU {sku}: {e}")
            if attempt == MAX_PARSE_RETRIES:
                raise RuntimeError(
                    f"Failed to extract ProductRecord for {sku} after {MAX_PARSE_RETRIES} retries. "
                    f"Last error: {e}"
                ) from e
            messages.append({
                "role": "user",
                "content": (
                    f"An unexpected error occurred:\n{e}\n\n"
                    f"Please return a corrected JSON object that matches the ProductRecord schema exactly."
                ),
            })


def run_enrichment_pipeline(
    client: anthropic.Anthropic,
    raw_products: Optional[dict[str, str]] = None,
) -> tuple[list[ProductRecord], dict]:
    """Run extract_product_record() over every raw vendor description.

    TODO (Phase 2):
    - Default raw_products to load_raw_products() when not provided
    - For each (sku, raw_description):
        * call extract_product_record(); on success append to a results list
        * on repeated ValidationError, log the failure (sku + error) instead
          of raising — this pipeline must not crash on one bad record
        * track a retry count per product (extract_product_record can return
          it, or you can count attempts here)
    - Print a final summary table: total processed, succeeded, failed, retried
    - Return (list[ProductRecord], summary_dict) where summary_dict has
      keys: "succeeded", "failed" (list of skus), "retried" (dict sku->count)
    """
    """
    This pipeline loads the raw catalogue, extracts ProductRecords, and
    gracefully handles per-product extraction failures without crashing.
    Returns both the list of successfully enriched records and a summary
    tracking successes, failures, and retry counts.
    """
    if raw_products is None:
        raw_products = load_raw_products()

    results: list[ProductRecord] = []
    failed_skus: list[str] = []

    total = len(raw_products)
    print(f"\n  Enriching {total} products...")

    for sku, raw_description in raw_products.items():
        try:
            record = extract_product_record(
                client=client,
                sku=sku,
                raw_description=raw_description,
            )
            results.append(record)
            print(f"    ✓ {sku}: {record.name}")
        except RuntimeError as e:
            # extract_product_record() raises RuntimeError after exhausting retries
            failed_skus.append(sku)
            print(f"    ✗ {sku}: {e}")

    succeeded = len(results)
    failed = len(failed_skus)

    summary = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "failed_skus": failed_skus,
    }

    print(f"\n  ┌─ Enrichment Pipeline Summary ─┐")
    print(f"  │ Total processed: {total}")
    print(f"  │ Succeeded:       {succeeded}")
    print(f"  │ Failed:          {failed}")
    print(f"  └─────────────────────────────┘")

    return results, summary



# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Multi-Turn Q&A Conversation (Day 2 skills)
# ═══════════════════════════════════════════════════════════════════════════════

class ProductConversationManager:
    """Maintains full message history for a multi-turn Q&A session, and
    tracks which product(s) have been discussed so far.

    TODO (Phase 3):
    - __init__(self, client, system): store client and system; initialise
      self.messages = [] and self.products_discussed: set[str] = set()
    - send(self, user_message, skus_mentioned=()) -> str:
        * update self.products_discussed with any skus_mentioned
        * append {"role": "user", "content": user_message}
        * call client.messages.create(model, max_tokens, system, messages)
        * append {"role": "assistant", "content": reply}   ← full list, not just .text
        * return the text reply
    - token_count(self) -> int:
        * use client.messages.count_tokens(model, system, messages)
        * return result.input_tokens
        * print a warning if this exceeds TOKEN_WARN_THRESHOLD
    - summarise_and_reset(self) -> str:
        * build a history_text string from self.messages
        * ask Claude to summarise in <=150 words, preserving which SKUs were
          discussed and any open questions
        * reset self.messages to [{"role":"user","content":"[Summary]\\n{summary}"}]
        * return the summary string
        * triggered by the caller once token_count() exceeds TOKEN_COMPACT_THRESHOLD
    """

    def __init__(self, client: anthropic.Anthropic, system: str) -> None:
        """Initialise the conversation manager with a client and system prompt.

        Args:
            client: Anthropic API client
            system: System prompt (typically QA_SYSTEM.format(product_context=...)
        """
        self.client = client
        self.system = system
        self.messages: list[dict] = []
        self.products_discussed: set[str] = set()

    def send(self, user_message: str, skus_mentioned: tuple[str, ...] = ()) -> str:
        """Send a user message and get an assistant reply in this conversation.

        Args:
            user_message: Customer query or input
            skus_mentioned: Tuple of SKU strings referenced in this turn

        Returns:
            The assistant's text reply (extracted from response.content)
        """
        # Track which products have been discussed
        self.products_discussed.update(skus_mentioned)

        # Add user turn
        self.messages.append({
            "role": "user",
            "content": user_message,
        })

        # Call Claude
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=self.system,
            messages=self.messages,
        )

        # Append full assistant content (list of blocks, not just text)
        self.messages.append({
            "role": "assistant",
            "content": response.content,
        })

        # Extract text reply from the response
        # response.content is a list of content blocks; find the text block
        reply_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                reply_text += block.text

        return reply_text

    def token_count(self) -> int:
        """Count tokens in the current conversation and warn if high.

        Returns:
            Total input tokens for the system + messages
        """
        result = self.client.messages.count_tokens(
            model=MODEL,
            system=self.system,
            messages=self.messages,
        )

        token_count = result.input_tokens

        # Warn if approaching the compaction threshold
        if token_count > TOKEN_WARN_THRESHOLD:
            print(
                f"\n  ⚠️  Token count ({token_count}) exceeds warning threshold ({TOKEN_WARN_THRESHOLD}). "
                f"Consider calling summarise_and_reset().\n"
            )

        return token_count

    def summarise_and_reset(self) -> str:
        """Summarise the conversation and reset message history for compaction.

        This is called when token_count() exceeds TOKEN_COMPACT_THRESHOLD,
        and compresses the full conversation into a brief summary to free tokens
        while preserving context.

        Returns:
            The generated summary string (≤150 words)
        """
        # Build a readable history from the messages list
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content'] if isinstance(msg['content'], str) else '[content]'}"
            for msg in self.messages
        ])

        # Ask Claude to summarise
        summary_response = self.client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=(
                "You are a conversational summariser. Create a brief summary ("
                "≤150 words) of the Q&A session below, preserving:\n"
                "- Which SKUs/products were discussed\n"
                "- Any open questions or next steps\n"
                "- Key customer preferences or concerns\n\n"
                "Be concise and factual."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarise this conversation:\n\n{history_text}",
                }
            ],
        )

        # Extract summary text
        summary = ""
        for block in summary_response.content:
            if hasattr(block, "text"):
                summary += block.text

        # Reset messages to just the summary
        self.messages = [
            {
                "role": "user",
                "content": f"[Prior conversation summary]\n{summary}",
            }
        ]

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Real-Time Tool Integration (Day 3 skills)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Mock tool implementations (provided — matches the mock DBs above) ──────────

def _check_inventory(sku: str) -> dict:
    """Look up live stock for a SKU. Returns an error dict for unknown SKUs."""
    result = INVENTORY_DB.get(sku)
    if not result:
        return {"error": f"SKU {sku} not found in inventory system."}
    return result


def _get_current_price(sku: str) -> dict:
    """Look up the live (possibly discounted) price for a SKU."""
    result = PRICE_DB.get(sku)
    if not result:
        return {"error": f"SKU {sku} not found in pricing system."}
    return result


def _fetch_vendor_spec(sku: str, spec_field: str) -> dict:
    """Look up one named spec field from the vendor spec sheet for a SKU."""
    key = (sku, spec_field.strip().lower().replace(" ", "_"))
    result = VENDOR_SPEC_DB.get(key)
    if not result:
        return {"field": spec_field, "value": None, "source": "not_found"}
    return result


TOOL_FN_MAP = {
    "check_inventory":    _check_inventory,
    "get_current_price":  _get_current_price,
    "fetch_vendor_spec":  _fetch_vendor_spec,
}


def build_qa_tools() -> list[dict]:
    """Return the list of tool definitions passed to client.messages.create().

    Three tools provide real-time data access:
    1. check_inventory() — live stock levels and warehouse location
    2. get_current_price() — live pricing with active discounts
    3. fetch_vendor_spec() — detailed product specifications from vendor sheets
    """
    return [
        {
            "name": "check_inventory",
            "description": (
                "Check live stock availability for a product SKU. Returns availability status, "
                "quantity on hand, and warehouse location. Call this when the customer asks "
                "about stock, availability, or when a product is in stock."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU (e.g., 'SKU-E001')",
                    },
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_current_price",
            "description": (
                "Get the current selling price (post-discount) and any active promotional offers. "
                "Returns INR price, discount percentage, and offer end date. Call this when "
                "the customer asks about price, cost, discount, or deal information."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU (e.g., 'SKU-E001')",
                    },
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
        {
            "name": "fetch_vendor_spec",
            "description": (
                "Retrieve a specific product specification from the vendor spec sheet. "
                "Call this when a detailed technical spec is missing from the catalogue context "
                "(e.g., RAM, ports, water resistance, certification). Provide the field name "
                "(e.g., 'ram', 'thunderbolt', 'water_resistance')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU (e.g., 'SKU-E001')",
                    },
                    "spec_field": {
                        "type": "string",
                        "description": (
                            "Name of the spec field to look up (e.g., 'ram', 'ports', "
                            "'water_resistance', 'warranty', 'isi_certification')"
                        ),
                    },
                },
                "required": ["sku", "spec_field"],
                "additionalProperties": False,
            },
        },
    ]


def run_qa_agentic_loop(
    client: anthropic.Anthropic,
    conversation_history: list[dict],
    tools: list[dict],
) -> list[dict]:
    """Run the manual agentic loop for the Q&A assistant: Claude calls tools,
    you execute them, loop until end_turn.

    TODO (Phase 4):
    - Start with messages = conversation_history (make a copy to be safe)
    - while True:
        * call client.messages.create(model, max_tokens, system=QA_SYSTEM,
          tools=tools, messages=messages)
        * if response.stop_reason == "end_turn": break
        * append {"role": "assistant", "content": response.content}   ← full list
        * for each tool_use block in response.content:
            - print the tool name and input
            - look up the function in TOOL_FN_MAP, call it with **block.input
            - set is_error = "error" in result
            - append tool_result to tool_results list
        * append {"role": "user", "content": tool_results}
    - Return the final messages list

    Key mistakes to avoid (same as the Finance case study):
    - Break on "end_turn", NOT on "tool_use"
    - Append response.content (the list), NOT response.content[0].text
    - Tool result content must be json.dumps(result) — a string, not a dict
    """
    messages = list(conversation_history)  # Make a defensive copy

    while True:
        # Call Claude with tools enabled
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=QA_SYSTEM,
            tools=tools,
            messages=messages,
        )

        # Always append the assistant's response (which may contain tool calls)
        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        # Check stop reason — if end_turn, we're done
        if response.stop_reason == "end_turn":
            break

        # Process tool calls in the response
        tool_results = []
        for block in response.content:
            # Only process tool_use blocks; ignore text blocks
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                print(f"    → Calling tool: {tool_name} with input: {tool_input}")

                # Look up and execute the tool function
                if tool_name not in TOOL_FN_MAP:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_fn = TOOL_FN_MAP[tool_name]
                        result = tool_fn(**tool_input)
                    except Exception as e:
                        result = {"error": f"Tool error: {e}"}

                # Determine if this result is an error
                is_error = "error" in result

                # Append the tool result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                })

        # If we have tool results, add them as a user message and continue the loop
        if tool_results:
            messages.append({
                "role": "user",
                "content": tool_results,
            })

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — RAG over Product Catalogue and Vendor Specs (Day 3-4 skills)
# ═══════════════════════════════════════════════════════════════════════════════

def build_product_index(records: list[ProductRecord]) -> object:
    """Build a Chroma collection over enriched products + vendor spec sheets.

    TODO (Phase 5):
    - import chromadb and voyageai lazily inside this function (so Phases 1-4
      run without those keys/packages configured)
    - Start a Chroma client: chromadb.PersistentClient(path="./chroma_data")
      (or EphemeralClient() for an in-memory index during development)
    - get_or_create_collection(CHROMA_COLLECTION_NAME) — pass an embedding_function
      that wraps voyageai.Client().embed(texts, model="voyage-3", input_type=...)
      or precompute embeddings yourself and pass them via collection.add(embeddings=...)
    - For each ProductRecord: build one text document (name + brand + category +
      key_features + specifications), and collection.add(
          ids=[sku], documents=[text], metadatas=[{"sku":.., "category":.., "brand":..}]
      )
    - Also read every *.md file under VENDOR_SPEC_DIR, chunk if needed, and
      collection.add(...) each with metadata {"sku": <sku>, "source": "vendor_spec"}
      (map filename -> sku, e.g. via a small dict or filename convention)
    - Return the populated collection
    """
    import chromadb
    from openai import OpenAI

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def openai_embedding_fn(texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI's embedding model."""
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    # Start Chroma client (use EphemeralClient for development)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add enriched ProductRecords to the collection
    print(f"\n  Building product index with {len(records)} enriched records...")
    for record in records:
        # Build a comprehensive document from the ProductRecord
        doc_text = (
            f"Product: {record.name}\n"
            f"Brand: {record.brand}\n"
            f"Category: {record.category}\n"
            f"Price: ₹{record.price_inr}\n"
            f"In Stock: {record.in_stock}\n"
            f"Features: {'; '.join(record.key_features) if record.key_features else 'N/A'}\n"
            f"Specifications: {'; '.join(f'{k}: {v}' for k, v in record.specifications.items()) if record.specifications else 'N/A'}"
        )

        # Embed and add to collection
        embedding = openai_embedding_fn([doc_text])[0]
        collection.add(
            ids=[record.sku],
            documents=[doc_text],
            embeddings=[embedding],
            metadatas=[{
                "sku": record.sku,
                "category": record.category,
                "brand": record.brand or "Unknown",
                "source": "catalogue",
                "in_stock": record.in_stock,
            }],
        )
        print(f"    ✓ Indexed {record.sku}: {record.name}")

    # Add vendor specification documents (if the directory exists)
    if VENDOR_SPEC_DIR.exists():
        print(f"\n  Indexing vendor specifications from {VENDOR_SPEC_DIR}...")
        spec_files = list(VENDOR_SPEC_DIR.glob("*.md"))
        for spec_file in spec_files:
            # Extract SKU from filename (e.g., "SKU-E001.md" → "SKU-E001")
            sku = spec_file.stem
            spec_text = spec_file.read_text(encoding="utf-8")

            # Embed and add to collection
            embedding = openai_embedding_fn([spec_text])[0]
            collection.add(
                ids=[f"{sku}_spec"],
                documents=[spec_text],
                embeddings=[embedding],
                metadatas=[{
                    "sku": sku,
                    "source": "vendor_spec",
                }],
            )
            print(f"    ✓ Indexed {sku} vendor specifications")
    else:
        print(f"\n  ℹ️  Vendor spec directory not found at {VENDOR_SPEC_DIR}")

    print(f"\n  ✓ Product index built with {collection.count()} documents")
    return collection


def retrieve_product_context(
    query: str,
    collection: object,
    category_filter: Optional[str] = None,
    top_k: int = TOP_K_CHUNKS,
) -> str:
    """Retrieve the top-k catalogue/vendor-spec chunks most relevant to query.

    TODO (Phase 5):
    - Embed the query with the same Voyage model used in build_product_index()
    - Call collection.query(query_embeddings=[...], n_results=top_k,
      where={"category": category_filter} if category_filter else None)
      — this is the metadata filter: apply it only when the customer has
      already specified a category (e.g. "I'm looking at laptops")
    - Format each result as a `[Product Context: SKU-XXX]` block followed by
      its text, joined by double newlines
    - Return the formatted string

    This string replaces {product_context} in QA_SYSTEM and is also where
    `"citations": {"enabled": true}` should be wired in on the message block
    that carries this context, so answers cite verifiable sources.
    """
    from openai import OpenAI

    # Initialize OpenAI client for embedding the query
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Embed the query
    query_response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    query_embedding = query_response.data[0].embedding

    # Query the collection with optional metadata filter
    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    # Format results into readable context blocks
    context_parts = []
    if results["documents"] and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            if doc:  # Skip empty documents
                metadata = results["metadatas"][0][i] if results["metadatas"] and i < len(results["metadatas"][0]) else {}
                sku = metadata.get("sku", "Unknown")
                source = metadata.get("source", "catalogue")
                context_parts.append(f"[Product Context: {sku} (from {source})]\n{doc}")

    # Join all chunks with double newlines for clarity
    product_context = "\n\n".join(context_parts) if context_parts else "(No product information available)"
    return product_context


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Evaluation (Day 4 skills)
# ═══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_JUDGE_SYSTEM = """
TODO (Phase 6): Write the faithfulness judge system prompt.

It should instruct Claude to:
- Score 1-5 how well an answer is supported by the provided product context
  (1 = hallucinated, 5 = fully grounded)
- Flag (in the reasoning) any warranty period, compatibility claim, or
  certification status stated in the answer but absent from the context
- Return JSON: {"score": <int>, "reasoning": "<str>"}
"""


def judge_faithfulness(client: anthropic.Anthropic, context: str, answer: str) -> dict:
    """Score a Q&A answer for faithfulness to the retrieved product context.

    TODO (Phase 6):
    - Build a user prompt combining context and answer
    - Call client.messages.create() with FAITHFULNESS_JUDGE_SYSTEM
    - Strip markdown fences from the response text before json.loads()
      Hint: re.search(r"```(?:json)?\\s*(\\{[\\s\\S]*?\\})\\s*```", text)
    - Return the parsed dict {"score": int, "reasoning": str}
    - On any parse error, return {"score": 0, "reasoning": "parse error: <raw text>"}
    """
    user_prompt = f"PRODUCT CONTEXT:\n{context}\n\nCUSTOMER ANSWER:\n{answer}\n\nScore this answer."
    response = client.messages.create(
        model=MODEL, max_tokens=256, system=FAITHFULNESS_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    response_text = "".join(block.text for block in response.content if hasattr(block, "text"))
    try:
        match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", response_text)
        json_str = match.group(1) if match else response_text.strip()
        return json.loads(json_str)
    except Exception as e:
        return {"score": 0, "reasoning": f"parse error: {str(e)[:50]}"}


def evaluate_enrichment_accuracy(
    records: list[ProductRecord],
    golden_set: list[dict] = ENRICHMENT_GOLDEN_SET,
) -> dict:
    """Compare enriched records against ENRICHMENT_GOLDEN_SET ground truth.

    Args:
        records: List of ProductRecord objects.
        golden_set: List of ground truth data for evaluation.

    Returns:
        A dictionary containing the evaluation results.
    """
    # Index records by sku
    record_dict = {record.sku: record for record in records}

    correct_fields = 0
    total_fields = 0
    hallucinated_specs = 0
    flagged_skus = []

    for golden in golden_set:
        sku = golden["sku"]
        ground_truth = golden["ground_truth"]

        if sku not in record_dict:
            hallucinated_specs += 1
            flagged_skus.append(sku)
            continue

        record = record_dict[sku]

        for field, value in ground_truth.items():
            if getattr(record, field) == value:
                correct_fields += 1
            else:
                hallucinated_specs += 1
                if sku not in flagged_skus:
                    flagged_skus.append(sku)
            total_fields += 1

    field_accuracy = correct_fields / total_fields if total_fields > 0 else 0.0

    return {
        "field_accuracy": field_accuracy,
        "hallucinated_specs": hallucinated_specs,
        "flagged_skus": flagged_skus,
    }


def log_eval_result(record: dict) -> None:
    """Append one evaluation result as a JSON line to EVAL_LOG_PATH.

    Args:
        record: Dictionary containing the evaluation result.
    """
    record["timestamp"] = datetime.datetime.utcnow().isoformat()
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOG_PATH.open(mode="a") as log_file:
        log_file.write(json.dumps(record) + "\n")


def run_evaluation(
    client: anthropic.Anthropic,
    records: list[ProductRecord],
    collection: object,
    tools: list[dict],
) -> None:
    """Run both evaluation tracks and print the combined report.

    Args:
        client: Anthropic API client.
        records: List of ProductRecord objects.
        collection: Chroma collection.
        tools: List of tool definitions.
    """
    # ENRICHMENT ACCURACY
    result = evaluate_enrichment_accuracy(records)
    log_eval_result({"track": "enrichment", **result})
    print("ENRICHMENT ACCURACY (5 golden products)")
    print(f"  Field Accuracy: {result['field_accuracy']:.2f}")
    print(f"  Hallucinated Specs: {result['hallucinated_specs']}")
    print(f"  Flagged SKUs: {result['flagged_skus']}")

    # Q&A FAITHFULNESS
    total_faith_score = 0
    for entry in QA_GOLDEN_SET:
        category_filter = None  # or infer one from the query
        context = retrieve_product_context(entry["query"], collection, category_filter)
        conversation_history = [
            {"role": "user", "content": entry["query"]},
        ]
        messages = run_qa_agentic_loop(client, conversation_history, tools)
        answer = "".join(block.text for block in messages[-1]["content"] if hasattr(block, "text"))
        faith = judge_faithfulness(client, context, answer)
        log_eval_result({"track": "qa", "query": entry["query"], **faith})
        total_faith_score += faith["score"]

    avg_faith_score = total_faith_score / len(QA_GOLDEN_SET) if QA_GOLDEN_SET else 0.0
    print("Q&A FAITHFULNESS (5 golden queries)")
    print(f"  Average Faithfulness Score: {avg_faith_score:.2f}")

    # Print an overall PASS/FAIL summary
    pass_threshold = 4.0
    if avg_faith_score >= pass_threshold and result["field_accuracy"] >= 0.9:
        print("\nPASS: Evaluation criteria met.")
    else:
        print("\nFAIL: Evaluation criteria not met.")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION — wire all phases together
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Phase 1: Initialise ────────────────────────────────────────────────────
    client = make_client()
    tools = build_qa_tools()        # Phase 4 — safe to call empty list until then

    # ── Phase 2: Enrich the full catalogue ────────────────────────────────────
    records, summary = run_enrichment_pipeline(client)
    print(f"\n  Enrichment summary: {summary}")

    # ── Phase 5: Build the RAG index ──────────────────────────────────────────
    # Comment this block out until Phase 5 is implemented:
    collection = build_product_index(records)
    # collection = None
    product_context = ""
    fallback = "I'm sorry, I don't have enough information to answer your question. Could you please provide more details?"
    # ── Run the test Q&A conversation end-to-end ──────────────────────────────
    print(f"\n{'='*60}")
    print("Running: Dell XPS 15 multi-turn Q&A test conversation")
    print("=" * 60)

    manager = ProductConversationManager(client, QA_SYSTEM.format(product_context=product_context, fallback=fallback))
    for turn in TEST_CONVERSATION:
        if collection is not None:
            product_context = retrieve_product_context(turn, collection, category_filter="electronics")
        reply = manager.send(turn, skus_mentioned=("SKU-E001",))
        print(f"\n  Customer: {turn}\n  Assistant: {reply}")

    # ── Phase 6: Run full evaluation ──────────────────────────────────────────
    # Uncomment when Phase 6 is ready:
    run_evaluation(client, records, collection, tools)


if __name__ == "__main__":
    main()
