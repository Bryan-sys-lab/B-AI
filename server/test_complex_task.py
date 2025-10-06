#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import asyncio
from providers.nim_adapter import NIMAdapter

async def test_complex_task():
    """Test which models can handle complex tasks"""

    # Complex task description
    description = """You are a senior full-stack engineer Aetherium. Your task is to generate production-grade code for an Aetherium-powered, agentic e-commerce meta-platform that integrates East African vendors, shops, and e-commerce sites.

🔑 Key Features to Implement

Customer Side:
- Chat-first shopping assistant (Aetherium-powered).
- Product comparison (price, vendor, delivery ETA, logistics, quality score).
- Live offers ticker (real-time streaming promotions tagged with vendor identity).
- Map pinning for delivery/pickup.
- Product categories (fashion, groceries, electronics, medical, etc), with support for dynamic expansion.

Vendor Side:
- Vendor portal (manage products, offers, logistics settings).
- Products must always be tagged by vendor.
- Vendors can create promotions and discounts.

Agentic System (event-driven microservices/workers):
- Price Watcher Agent → monitors product prices across vendors.
- Offer Curator Agent → prioritizes and streams offers.
- Fraud & Quality Agent → flags anomalies/counterfeit items.
- Category Evolution Agent → suggests new categories based on trends.
- Logistics Optimizer Agent → recommends delivery partners.
- Onboarding Assistant Agent → extracts product details from vendor uploads (text/image).
- Customer Service Copilot → Aetherium-driven FAQ + escalation.

🏗️ System Requirements
- Frontend: Next.js (TypeScript, Tailwind, ShadCN UI), chat-first UX, minimal pages.
- Backend: Python (FastAPI) microservices.
- Data Layer: PostgreSQL (core data), Elasticsearch (search), Redis (cache + pub/sub), S3 storage (images).
- Realtime: WebSockets/Event streams (offers, chat, order tracking).
- Aetherium Gateway: LLM for chat & recommendations, embeddings for semantic search.
- Payments: Mobile money integration (M-Pesa, Airtel).
- Security: Vendor KYC, RBAC, audit logs, API rate limiting.

📂 Expected Output Format
Generate code for all services and frontend in the proper directory structure with database schema, event schema, and OpenAPI specs."""

    print("Testing complex e-commerce task with different models...")
    print(f"Task description length: {len(description)} characters")

    # Test different models
    models_to_test = [
        "nvidia/nemotron-mini-4b-instruct",
        "nvidia/llama-3.1-nemotron-51b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/usdcode-llama-3.1-70b-instruct"
    ]

    messages = [
        {"role": "system", "content": "You are a senior full-stack engineer. Generate complete, working code based on user requests."},
        {"role": "user", "content": description}
    ]

    for model in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")

        try:
            adapter = NIMAdapter(role="builders")
            # Override the model
            adapter.default_model = model

            response = adapter.call_model(messages, temperature=0.7)
            print(f"✓ SUCCESS - Model {model}")
            print(f"  Response length: {len(response.text)} characters")
            print(f"  Tokens used: {response.tokens}")
            print(f"  Latency: {response.latency_ms}ms")
            print(f"  First 200 chars: {response.text[:200]}...")

        except Exception as e:
            print(f"✗ FAILED - Model {model}")
            print(f"  Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_complex_task())