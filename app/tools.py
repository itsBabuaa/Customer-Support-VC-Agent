"""Agent tools: RAG search, order lookup, product info, escalation."""

import logging


from livekit.agents.llm import function_tool


from app.rag import KnowledgeRAG
from app.dummy_apis import (

    lookup_order, lookup_by_phone, lookup_by_email,
    get_product_info, list_products,
)

from app.config import RAG_SOURCE


logger = logging.getLogger("voice-agent.tools")


_rag: KnowledgeRAG | None = None



def _get_rag() -> KnowledgeRAG:

    global _rag

    if _rag is None:

        logger.info("Lazy-loading RAG index from %s", RAG_SOURCE)

        _rag = KnowledgeRAG(RAG_SOURCE)

        logger.info("Knowledge base ready: %d chunks", len(_rag.chunks))
    return _rag



@function_tool()

async def search_knowledge(query: str) -> str:

    """Search the knowledge base for laptop specs, policies, warranty info.

    Pass the customer's exact question. Returns relevant info.

    Call EXACTLY ONCE per user message."""

    logger.info("[TOOL] search_knowledge: %s", query)

    result = _get_rag().search(query, top_k=3)

    logger.info("[TOOL] result: %s", result[:150])

    return result[:800]



@function_tool()

async def check_order(order_id: str) -> str:

    """Look up order by ID (e.g. NLT-10001). Returns status, items, tracking."""

    logger.info("[TOOL] check_order: %s", order_id)
    return lookup_order(order_id)



@function_tool()

async def check_order_by_phone(phone_number: str) -> str:

    """Look up orders by customer phone number."""

    logger.info("[TOOL] check_order_by_phone: %s", phone_number)

    return lookup_by_phone(phone_number)



@function_tool()

async def check_order_by_email(email: str) -> str:

    """Look up orders by customer email address."""

    logger.info("[TOOL] check_order_by_email: %s", email)

    return lookup_by_email(email)



@function_tool()

async def get_laptop_specs(sku: str) -> str:

    """Get detailed specs for a laptop by SKU (PRO-16, AIR-14, STU-15, GAM-17)."""

    logger.info("[TOOL] get_laptop_specs: %s", sku)
    return get_product_info(sku)



@function_tool()

async def list_all_laptops() -> str:

    """List all available laptop models with prices."""

    logger.info("[TOOL] list_all_laptops")
    return list_products()



@function_tool()

async def escalate_to_human(reason: str) -> str:

    """Transfer the call to a human agent. Use when:

    - Customer explicitly asks for a human

    - Issue is too complex (hardware defect, refund dispute)

    - Customer is frustrated after multiple attempts"""

    logger.info("[TOOL] escalate_to_human: %s", reason)
    return (

        f"ESCALATION INITIATED — Reason: {reason}. "

        "A human agent will be connected shortly. "

        "Please reassure the customer that help is on the way."
    )



ALL_TOOLS = [

    search_knowledge,
    check_order,

    check_order_by_phone,

    check_order_by_email,
    get_laptop_specs,
    list_all_laptops,
    escalate_to_human,

]

