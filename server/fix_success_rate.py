#!/usr/bin/env python3
"""
Fix success rate defaults in database.
Sets success_rate to 0.0 for providers and agents with 0 total_requests/tasks_completed.
"""

import asyncio
from sqlalchemy import text, update
from dotenv import load_dotenv
load_dotenv()

from orchestrator.database import async_session, ProviderMetrics, AgentMetrics

async def fix_success_rates():
    """Update success rates for providers and agents with no activity."""
    async with async_session() as session:
        # Fix provider metrics
        result = await session.execute(
            update(ProviderMetrics)
            .where(ProviderMetrics.total_requests == 0)
            .where(ProviderMetrics.success_rate == 100.0)
            .values(success_rate=0.0)
        )
        provider_count = result.rowcount
        print(f"Updated {provider_count} provider metrics")

        # Fix agent metrics
        result = await session.execute(
            update(AgentMetrics)
            .where(AgentMetrics.tasks_completed == 0)
            .where(AgentMetrics.success_rate == 100.0)
            .values(success_rate=0.0)
        )
        agent_count = result.rowcount
        print(f"Updated {agent_count} agent metrics")

        await session.commit()
        print("Success rate fix completed")

if __name__ == "__main__":
    asyncio.run(fix_success_rates())