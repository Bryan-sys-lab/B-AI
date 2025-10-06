import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class RelationshipTracker:
    def __init__(self):
        self.relationships_file = os.path.join(os.path.dirname(__file__), "relationships.json")
        self.relationships = defaultdict(list)  # agent -> list of relationships
        self.load_relationships()

    async def initialize(self):
        """Initialize the relationship tracker"""
        logger.info("Initializing Relationship Tracker")
        self.load_relationships()

    def load_relationships(self):
        """Load relationships from file"""
        try:
            if os.path.exists(self.relationships_file):
                with open(self.relationships_file, 'r') as f:
                    data = json.load(f)
                    self.relationships = defaultdict(list, data)
                logger.info(f"Loaded {sum(len(v) for v in self.relationships.values())} relationships")
            else:
                logger.info("No existing relationships file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading relationships: {e}")
            self.relationships = defaultdict(list)

    def save_relationships(self):
        """Save relationships to file"""
        try:
            with open(self.relationships_file, 'w') as f:
                json.dump(dict(self.relationships), f, indent=2)
            logger.debug("Relationships saved to file")
        except Exception as e:
            logger.error(f"Error saving relationships: {e}")

    async def store_relationship(self, source_agent: str, target_agent: str,
                               relationship_type: str, strength: float,
                               context: Optional[Dict[str, Any]] = None) -> str:
        """Store a relationship between agents"""
        try:
            relationship_id = f"{source_agent}_{target_agent}_{relationship_type}_{asyncio.get_event_loop().time()}"

            relationship = {
                "id": relationship_id,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "relationship_type": relationship_type,
                "strength": strength,
                "context": context or {},
                "timestamp": asyncio.get_event_loop().time(),
                "observations": 1
            }

            # Check if relationship already exists
            existing_relationships = self.relationships[source_agent]
            existing = None
            for rel in existing_relationships:
                if (rel["target_agent"] == target_agent and
                    rel["relationship_type"] == relationship_type):
                    existing = rel
                    break

            if existing:
                # Update existing relationship
                existing["strength"] = (existing["strength"] + strength) / 2  # Average
                existing["observations"] += 1
                existing["last_updated"] = asyncio.get_event_loop().time()
                if context:
                    existing["context"].update(context)
                relationship_id = existing["id"]
                logger.info(f"Updated existing relationship: {source_agent} -> {target_agent}")
            else:
                # Add new relationship
                self.relationships[source_agent].append(relationship)
                logger.info(f"Created new relationship: {source_agent} -> {target_agent} ({relationship_type})")

            self.save_relationships()
            return relationship_id

        except Exception as e:
            logger.error(f"Error storing relationship: {e}")
            raise

    async def get_relationships(self, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relationships for an agent or all relationships"""
        try:
            if agent_name:
                return self.relationships.get(agent_name, [])
            else:
                # Return all relationships
                all_relationships = []
                for agent, rels in self.relationships.items():
                    all_relationships.extend(rels)
                return all_relationships
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []

    async def get_related_agents(self, agent_name: str, relationship_type: Optional[str] = None,
                               min_strength: float = 0.0) -> List[Dict[str, Any]]:
        """Get agents related to the given agent"""
        try:
            relationships = self.relationships.get(agent_name, [])
            related_agents = []

            for rel in relationships:
                if rel["strength"] >= min_strength:
                    if relationship_type is None or rel["relationship_type"] == relationship_type:
                        related_agents.append({
                            "agent": rel["target_agent"],
                            "relationship_type": rel["relationship_type"],
                            "strength": rel["strength"],
                            "observations": rel.get("observations", 1),
                            "context": rel.get("context", {})
                        })

            # Also check reverse relationships (who considers this agent as target)
            for source_agent, rels in self.relationships.items():
                if source_agent != agent_name:
                    for rel in rels:
                        if rel["target_agent"] == agent_name and rel["strength"] >= min_strength:
                            if relationship_type is None or rel["relationship_type"] == relationship_type:
                                related_agents.append({
                                    "agent": source_agent,
                                    "relationship_type": f"reverse_{rel['relationship_type']}",
                                    "strength": rel["strength"],
                                    "observations": rel.get("observations", 1),
                                    "context": rel.get("context", {})
                                })

            logger.info(f"Found {len(related_agents)} related agents for {agent_name}")
            return related_agents

        except Exception as e:
            logger.error(f"Error getting related agents: {e}")
            return []

    async def analyze_relationships(self) -> Dict[str, Any]:
        """Analyze relationship patterns"""
        try:
            analysis = {
                "total_relationships": 0,
                "unique_agent_pairs": set(),
                "relationship_types": defaultdict(int),
                "strongest_relationships": [],
                "most_connected_agents": []
            }

            all_relationships = []
            for agent, rels in self.relationships.items():
                all_relationships.extend(rels)

            analysis["total_relationships"] = len(all_relationships)

            # Count relationship types and find unique pairs
            for rel in all_relationships:
                analysis["relationship_types"][rel["relationship_type"]] += 1
                pair = tuple(sorted([rel["source_agent"], rel["target_agent"]]))
                analysis["unique_agent_pairs"].add(pair)

            # Find strongest relationships
            sorted_rels = sorted(all_relationships, key=lambda x: x["strength"], reverse=True)
            analysis["strongest_relationships"] = sorted_rels[:5]

            # Find most connected agents
            agent_connections = defaultdict(int)
            for rel in all_relationships:
                agent_connections[rel["source_agent"]] += 1
                agent_connections[rel["target_agent"]] += 1

            sorted_agents = sorted(agent_connections.items(), key=lambda x: x[1], reverse=True)
            analysis["most_connected_agents"] = sorted_agents[:5]

            logger.info(f"Completed relationship analysis: {analysis['total_relationships']} relationships analyzed")
            return dict(analysis)  # Convert defaultdict to regular dict

        except Exception as e:
            logger.error(f"Error analyzing relationships: {e}")
            return {"error": str(e)}

    async def get_collaboration_suggestions(self, task_description: str) -> List[Dict[str, Any]]:
        """Suggest agent collaborations based on task and relationship history"""
        try:
            suggestions = []

            # Analyze task to identify potential agent needs
            task_lower = task_description.lower()
            potential_needs = []

            if "debug" in task_lower or "fix" in task_lower:
                potential_needs.append("debugger")
            if "implement" in task_lower or "code" in task_lower:
                potential_needs.append("fix_implementation")
            if "review" in task_lower or "quality" in task_lower:
                potential_needs.append("review")
            if "test" in task_lower:
                potential_needs.append("testing")
            if "deploy" in task_lower:
                potential_needs.append("deployment")

            # Find agents that have successfully collaborated on similar tasks
            for need in potential_needs:
                related_agents = await self.get_related_agents(need, "collaborates_with", 0.5)
                for related in related_agents:
                    suggestions.append({
                        "primary_agent": need,
                        "suggested_collaborator": related["agent"],
                        "relationship_strength": related["strength"],
                        "reason": f"Based on successful past collaborations"
                    })

            logger.info(f"Generated {len(suggestions)} collaboration suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"Error getting collaboration suggestions: {e}")
            return []

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about relationships"""
        try:
            analysis = await self.analyze_relationships()

            return {
                "total_relationships": analysis.get("total_relationships", 0),
                "unique_agent_pairs": len(analysis.get("unique_agent_pairs", set())),
                "relationship_types": dict(analysis.get("relationship_types", {})),
                "most_connected_agents": analysis.get("most_connected_agents", [])
            }
        except Exception as e:
            logger.error(f"Error getting relationship stats: {e}")
            return {"error": str(e)}