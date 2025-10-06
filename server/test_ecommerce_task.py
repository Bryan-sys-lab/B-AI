#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/su/Aetherium/B1.0')
sys.path.insert(0, '/home/su/Aetherium/B1.0/codeagent_venv/lib/python3.13/site-packages')

import os
import asyncio
from agents.fix_implementation.main import ExecuteRequest, execute_task

async def test_ecommerce_generation():
    """Test the e-commerce platform generation task with different roles"""

    # Sample task description
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

    print("Testing e-commerce platform generation task...")
    print(f"Task description length: {len(description)} characters")

    # Set up artifacts directory
    artifacts_dir = "/tmp/test_artifacts"
    os.environ["ARTIFACTS_DIR"] = artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        # Create ExecuteRequest object
        request = ExecuteRequest(description=description)

        # Execute the task
        result = await execute_task(request)

        print("\n=== EXECUTION RESULT ===")
        print(f"Success: {result.get('success', False)}")
        print(f"Tokens used: {result.get('tokens', 0)}")
        print(f"Latency: {result.get('latency_ms', 0)}ms")
        print(f"Is creation task: {result.get('is_creation_task', False)}")

        # Check for artifacts
        artifacts_saved = result.get('artifacts_saved', [])
        print(f"\nArtifacts saved: {artifacts_saved}")

        if artifacts_saved:
            print("\n=== ARTIFACTS CONTENT ===")
            for artifact in artifacts_saved:
                artifact_path = os.path.join(artifacts_dir, artifact)
                if os.path.exists(artifact_path):
                    print(f"\n--- {artifact} ---")
                    try:
                        with open(artifact_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Show first 500 chars
                            print(content[:500] + ("..." if len(content) > 500 else ""))
                    except Exception as e:
                        print(f"Error reading {artifact}: {e}")

        # Show structured response
        structured = result.get('structured', {})
        if structured:
            print("\n=== STRUCTURED RESPONSE ===")
            print(f"Description: {structured.get('description', '')[:200]}...")
            files = structured.get('files', {})
            print(f"Files generated: {len(files)}")
            for file_path in list(files.keys())[:5]:  # Show first 5 files
                print(f"  - {file_path}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")

        return result

    except Exception as e:
        print(f"Task execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_ecommerce_generation())