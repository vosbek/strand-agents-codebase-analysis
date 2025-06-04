#!/usr/bin/env python3
"""
Codebase Analysis Agent using AWS Strands
Focuses on legacy code analysis, business logic extraction, and modernization planning
"""

import os
import json
import ast
import javalang
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import boto3
from strands_agents import Agent, BedrockClient
from strands_agents.tools import tool
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class BusinessLogicEntity:
    """Represents a piece of business logic found in code"""
    name: str
    type: str  # action, service, validator, etc.
    file_path: str
    line_range: tuple
    description: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    risk_level: str  # low, medium, high
    complexity_score: int


@dataclass
class DataFlowNode:
    """Represents a node in the data flow graph"""
    id: str
    name: str
    type: str  # database, service, controller, etc.
    file_path: str
    business_logic: Optional[str] = None


class CodebaseAnalyzer:
    """Core analyzer for extracting business logic from codebases"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.business_entities = []
        self.data_flow_graph = nx.DiGraph()
        
    def analyze_java_file(self, file_path: Path) -> List[BusinessLogicEntity]:
        """Analyze a single Java file for business logic"""
        entities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse Java using javalang
            tree = javalang.parse.parse(content)
            
            # Look for Struts Action classes
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                if self._is_struts_action(node):
                    entity = self._extract_action_business_logic(node, file_path, content)
                    if entity:
                        entities.append(entity)
                        
            # Look for Spring components
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                if self._is_spring_component(node):
                    entity = self._extract_spring_business_logic(node, file_path, content)
                    if entity:
                        entities.append(entity)
                        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return entities
    
    def _is_struts_action(self, node) -> bool:
        """Check if class is a Struts Action"""
        if not node.extends:
            return False
        return 'Action' in str(node.extends.name)
    
    def _is_spring_component(self, node) -> bool:
        """Check if class has Spring annotations"""
        if not hasattr(node, 'annotations') or not node.annotations:
            return False
        
        spring_annotations = ['Service', 'Component', 'Controller', 'Repository']
        for annotation in node.annotations:
            if annotation.name in spring_annotations:
                return True
        return False
    
    def _extract_action_business_logic(self, node, file_path, content) -> Optional[BusinessLogicEntity]:
        """Extract business logic from Struts Action class"""
        # Look for execute method and other business methods
        business_methods = []
        dependencies = []
        
        for method in node.methods:
            if method.name in ['execute', 'perform'] or 'business' in method.name.lower():
                business_methods.append(method.name)
            
            # Check for service calls, DAO calls, etc.
            if hasattr(method, 'body') and method.body:
                deps = self._extract_dependencies_from_method(method)
                dependencies.extend(deps)
        
        if business_methods:
            return BusinessLogicEntity(
                name=node.name,
                type="struts_action",
                file_path=str(file_path),
                line_range=(node.position.line if hasattr(node, 'position') else 0, 0),
                description=f"Struts Action class with business methods: {', '.join(business_methods)}",
                inputs=self._extract_form_inputs(node),
                outputs=self._extract_action_outputs(node),
                dependencies=dependencies,
                risk_level=self._assess_risk_level(node, dependencies),
                complexity_score=self._calculate_complexity(node)
            )
        return None
    
    def _extract_spring_business_logic(self, node, file_path, content) -> Optional[BusinessLogicEntity]:
        """Extract business logic from Spring components"""
        # Similar to Struts but look for different patterns
        business_methods = []
        dependencies = []
        
        for method in node.methods:
            if not self._is_infrastructure_method(method.name):
                business_methods.append(method.name)
                if hasattr(method, 'body') and method.body:
                    deps = self._extract_dependencies_from_method(method)
                    dependencies.extend(deps)
        
        if business_methods:
            return BusinessLogicEntity(
                name=node.name,
                type="spring_component",
                file_path=str(file_path),
                line_range=(node.position.line if hasattr(node, 'position') else 0, 0),
                description=f"Spring component with business methods: {', '.join(business_methods)}",
                inputs=[],  # Would need more sophisticated analysis
                outputs=[],
                dependencies=dependencies,
                risk_level=self._assess_risk_level(node, dependencies),
                complexity_score=self._calculate_complexity(node)
            )
        return None
    
    def _extract_dependencies_from_method(self, method) -> List[str]:
        """Extract service/DAO dependencies from method calls"""
        dependencies = []
        
        if not hasattr(method, 'body') or not method.body:
            return dependencies
        
        # Look for method invocations in the method body
        for path, node in method.body.filter(javalang.tree.MethodInvocation):
            if hasattr(node, 'member'):
                method_name = node.member
                # Look for common service/DAO patterns
                if any(pattern in method_name.lower() for pattern in 
                       ['service', 'dao', 'repository', 'manager', 'helper', 'util']):
                    dependencies.append(method_name)
                
                # Look for database-related calls
                if any(pattern in method_name.lower() for pattern in 
                       ['execute', 'query', 'update', 'insert', 'delete', 'select']):
                    dependencies.append(f"database_operation_{method_name}")
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_form_inputs(self, node) -> List[str]:
        """Extract form input parameters from Action class"""
        inputs = []
        
        # Look for ActionForm parameters in methods
        if hasattr(node, 'methods'):
            for method in node.methods:
                if hasattr(method, 'parameters') and method.parameters:
                    for param in method.parameters:
                        param_type = str(param.type.name) if hasattr(param.type, 'name') else str(param.type)
                        if 'form' in param_type.lower() or 'actionform' in param_type.lower():
                            inputs.append(param_type)
        
        # Look for request.getParameter calls (would need more sophisticated parsing)
        inputs.extend(["request_parameters"])  # Placeholder - would need deeper analysis
        
        return inputs
    
    def _extract_action_outputs(self, node) -> List[str]:
        """Extract output/forward mappings from Action"""
        outputs = []
        
        # Look for ActionForward returns and mapping.findForward calls
        if hasattr(node, 'methods'):
            for method in node.methods:
                if hasattr(method, 'body') and method.body:
                    # Look for return statements and forward calls
                    for path, stmt_node in method.body.filter(javalang.tree.ReturnStatement):
                        if hasattr(stmt_node, 'expression'):
                            outputs.append("action_forward")
                    
                    # Look for mapping.findForward calls
                    for path, call_node in method.body.filter(javalang.tree.MethodInvocation):
                        if hasattr(call_node, 'member') and 'forward' in call_node.member.lower():
                            outputs.append("forward_mapping")
        
        # Common Struts outputs
        outputs.extend(["success", "failure", "input"])  # Standard Struts forwards
        
        return list(set(outputs))
    
    def _is_infrastructure_method(self, method_name: str) -> bool:
        """Check if method is infrastructure vs business logic"""
        infrastructure_patterns = ['get', 'set', 'init', 'destroy', 'toString', 'equals', 'hashCode']
        return any(pattern in method_name.lower() for pattern in infrastructure_patterns)
    
    def _assess_risk_level(self, node, dependencies) -> str:
        """Assess migration risk based on complexity and dependencies"""
        risk_score = 0
        
        # More methods = higher risk
        if hasattr(node, 'methods'):
            risk_score += len(node.methods) * 2
        
        # More dependencies = higher risk
        risk_score += len(dependencies) * 3
        
        # Check for complex patterns
        if len(dependencies) > 10:
            risk_score += 20
        
        if risk_score < 20:
            return "low"
        elif risk_score < 50:
            return "medium"
        else:
            return "high"
    
    def _calculate_complexity(self, node) -> int:
        """Calculate complexity score for the class"""
        complexity = 0
        if hasattr(node, 'methods'):
            complexity += len(node.methods) * 2
        if hasattr(node, 'fields'):
            complexity += len(node.fields)
        return complexity


# Strands Agent Tools

@tool
def analyze_codebase_structure(repo_path: str) -> Dict[str, Any]:
    """
    Analyze the overall structure of a codebase to understand its architecture.
    
    Args:
        repo_path: Path to the local repository
        
    Returns:
        Dictionary containing codebase structure analysis
    """
    analyzer = CodebaseAnalyzer(repo_path)
    repo_path_obj = Path(repo_path)
    
    # Find all relevant files
    java_files = list(repo_path_obj.rglob("*.java"))
    js_files = list(repo_path_obj.rglob("*.js"))
    perl_files = list(repo_path_obj.rglob("*.pl")) + list(repo_path_obj.rglob("*.pm"))
    
    structure = {
        "total_files": len(java_files) + len(js_files) + len(perl_files),
        "java_files": len(java_files),
        "javascript_files": len(js_files),
        "perl_files": len(perl_files),
        "estimated_loc": 0,  # Would calculate actual LOC
        "framework_indicators": {
            "struts": len([f for f in java_files if "struts" in str(f).lower()]),
            "spring": len([f for f in java_files if "spring" in str(f).lower()]),
            "angular": len([f for f in js_files if "angular" in str(f).lower()]),
        }
    }
    
    return structure


@tool
def analyze_struts_configuration(repo_path: str) -> Dict[str, Any]:
    """
    Analyze Struts configuration files to understand application structure.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Analysis of Struts configuration and mappings
    """
    repo_path_obj = Path(repo_path)
    
    # Find Struts configuration files
    config_files = []
    config_files.extend(list(repo_path_obj.rglob("struts-config.xml")))
    config_files.extend(list(repo_path_obj.rglob("validation.xml")))
    config_files.extend(list(repo_path_obj.rglob("struts.xml")))
    
    analysis = {
        "config_files_found": len(config_files),
        "action_mappings": [],
        "form_beans": [],
        "forwards": [],
        "validation_rules": [],
        "message_resources": []
    }
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple regex-based extraction (would use XML parser in production)
            import re
            
            # Extract action mappings
            action_pattern = r'<action[^>]*path="([^"]*)"[^>]*type="([^"]*)"'
            actions = re.findall(action_pattern, content)
            for path, action_class in actions:
                analysis["action_mappings"].append({
                    "path": path,
                    "action_class": action_class,
                    "config_file": str(config_file.relative_to(repo_path_obj))
                })
            
            # Extract form beans
            form_pattern = r'<form-bean[^>]*name="([^"]*)"[^>]*type="([^"]*)"'
            forms = re.findall(form_pattern, content)
            for name, form_class in forms:
                analysis["form_beans"].append({
                    "name": name,
                    "form_class": form_class,
                    "config_file": str(config_file.relative_to(repo_path_obj))
                })
            
            # Extract global forwards
            forward_pattern = r'<forward[^>]*name="([^"]*)"[^>]*path="([^"]*)"'
            forwards = re.findall(forward_pattern, content)
            for name, path in forwards:
                analysis["forwards"].append({
                    "name": name,
                    "path": path,
                    "config_file": str(config_file.relative_to(repo_path_obj))
                })
                
        except Exception as e:
            print(f"Error parsing {config_file}: {e}")
            continue
    
    # Add summary statistics
    analysis["summary"] = {
        "total_actions": len(analysis["action_mappings"]),
        "total_forms": len(analysis["form_beans"]),
        "total_forwards": len(analysis["forwards"]),
        "unique_action_classes": len(set(action["action_class"] for action in analysis["action_mappings"]))
    }
    
    return analysis


@tool
def extract_business_logic(repo_path: str, file_pattern: str = "*.java") -> List[Dict[str, Any]]:
    """
    Extract business logic entities from codebase files.
    
    Args:
        repo_path: Path to the local repository
        file_pattern: Pattern to match files (default: *.java)
        
    Returns:
        List of business logic entities found
    """
    analyzer = CodebaseAnalyzer(repo_path)
    repo_path_obj = Path(repo_path)
    
    all_entities = []
    
    # Process Java files
    java_files = list(repo_path_obj.rglob(file_pattern))
    
    for java_file in java_files[:50]:  # Limit for MVP
        try:
            entities = analyzer.analyze_java_file(java_file)
            for entity in entities:
                all_entities.append({
                    "name": entity.name,
                    "type": entity.type,
                    "file_path": entity.file_path,
                    "description": entity.description,
                    "risk_level": entity.risk_level,
                    "complexity_score": entity.complexity_score,
                    "dependencies": entity.dependencies
                })
        except Exception as e:
            print(f"Error processing {java_file}: {e}")
            continue
    
    return all_entities


@tool
def build_data_flow_map(repo_path: str, focus_entity: str = None) -> Dict[str, Any]:
    """
    Build a data flow map showing how data moves through the system.
    
    Args:
        repo_path: Path to the local repository
        focus_entity: Optional specific entity to focus the analysis on
        
    Returns:
        Data flow map as nodes and edges
    """
    analyzer = CodebaseAnalyzer(repo_path)
    repo_path_obj = Path(repo_path)
    
    nodes = []
    edges = []
    
    # Extract business entities first
    java_files = list(repo_path_obj.rglob("*.java"))
    business_entities = []
    
    for java_file in java_files[:100]:  # Limit for performance
        try:
            entities = analyzer.analyze_java_file(java_file)
            business_entities.extend(entities)
        except Exception:
            continue
    
    # Create nodes from discovered entities
    for entity in business_entities:
        nodes.append({
            "id": entity.name,
            "name": entity.name,
            "type": entity.type,
            "file_path": entity.file_path,
            "business_logic": entity.description
        })
        
        # Create edges based on dependencies
        for dep in entity.dependencies:
            edges.append({
                "from": entity.name,
                "to": dep,
                "type": "dependency",
                "description": f"{entity.name} depends on {dep}"
            })
    
    # Add common architectural layers if no specific entities found
    if not nodes:
        nodes = [
            {"id": "web_layer", "name": "Web Layer (JSPs)", "type": "presentation"},
            {"id": "action_layer", "name": "Struts Actions", "type": "controller"},
            {"id": "business_layer", "name": "Business Services", "type": "service"},
            {"id": "data_layer", "name": "Data Access Objects", "type": "persistence"},
            {"id": "database", "name": "Oracle Database", "type": "database"}
        ]
        edges = [
            {"from": "web_layer", "to": "action_layer", "type": "http_request"},
            {"from": "action_layer", "to": "business_layer", "type": "service_call"},
            {"from": "business_layer", "to": "data_layer", "type": "data_access"},
            {"from": "data_layer", "to": "database", "type": "sql_query"}
        ]
    
    flow_map = {
        "nodes": nodes,
        "edges": edges,
        "focus_entity": focus_entity,
        "metadata": {
            "total_entities": len(business_entities),
            "analyzed_files": min(len(java_files), 100)
        }
    }
    
    return flow_map


@tool
def generate_knowledge_graph(business_entities: List[Dict], output_path: str = "/tmp/knowledge_graph.json") -> str:
    """
    Generate a knowledge graph from business entities.
    
    Args:
        business_entities: List of business logic entities
        output_path: Where to save the knowledge graph
        
    Returns:
        Path to generated knowledge graph file
    """
    # Build knowledge graph structure
    graph_data = {
        "nodes": [],
        "relationships": [],
        "metadata": {
            "generated_at": "2025-06-03",
            "entity_count": len(business_entities)
        }
    }
    
    # Add nodes for each business entity
    for entity in business_entities:
        graph_data["nodes"].append({
            "id": entity["name"],
            "label": entity["name"],
            "type": entity["type"],
            "file_path": entity["file_path"],
            "risk_level": entity["risk_level"],
            "complexity": entity["complexity_score"],
            "properties": {
                "description": entity["description"]
            }
        })
        
        # Add relationships based on dependencies
        for dep in entity.get("dependencies", []):
            graph_data["relationships"].append({
                "from": entity["name"],
                "to": dep,
                "type": "depends_on"
            })
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    return output_path


@tool
def create_migration_documentation(business_entities: List[Dict], output_format: str = "markdown") -> str:
    """
    Create documentation for migration planning.
    
    Args:
        business_entities: List of business logic entities
        output_format: Format for documentation (markdown, json)
        
    Returns:
        Generated documentation content
    """
    if output_format == "markdown":
        doc = "# Business Logic Migration Documentation\n\n"
        doc += "## Overview\n\n"
        doc += f"This document contains {len(business_entities)} business logic entities identified for migration.\n\n"
        
        # Group by risk level
        risk_groups = {"low": [], "medium": [], "high": []}
        for entity in business_entities:
            risk_level = entity.get("risk_level", "medium")
            risk_groups[risk_level].append(entity)
        
        for risk_level in ["low", "medium", "high"]:
            entities = risk_groups[risk_level]
            if entities:
                doc += f"## {risk_level.title()} Risk Entities ({len(entities)})\n\n"
                for entity in entities:
                    doc += f"### {entity['name']}\n"
                    doc += f"- **Type**: {entity['type']}\n"
                    doc += f"- **File**: `{entity['file_path']}`\n"
                    doc += f"- **Description**: {entity['description']}\n"
                    doc += f"- **Complexity Score**: {entity['complexity_score']}\n"
                    if entity.get('dependencies'):
                        doc += f"- **Dependencies**: {', '.join(entity['dependencies'])}\n"
                    doc += "\n"
        
        return doc
    else:
        return json.dumps(business_entities, indent=2)


@tool
def search_business_logic(repo_path: str, query: str, context_lines: int = 5) -> List[Dict[str, Any]]:
    """
    Search for specific business logic patterns or keywords in the codebase.
    
    Args:
        repo_path: Path to the repository
        query: Search query (business term, method name, etc.)
        context_lines: Number of context lines to include
        
    Returns:
        List of search results with context and relevance scoring
    """
    repo_path_obj = Path(repo_path)
    results = []
    
    # Enhanced search patterns for business logic
    business_keywords = ['business', 'calculate', 'process', 'validate', 'transform', 'rule', 'logic']
    query_terms = query.lower().split()
    
    # Search in Java files
    java_files = list(repo_path_obj.rglob("*.java"))
    
    for java_file in java_files[:200]:  # Increase limit but still manageable
        try:
            with open(java_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                line_lower = line.lower()
                
                # Check if query terms are in the line
                if any(term in line_lower for term in query_terms):
                    start_line = max(0, i - context_lines)
                    end_line = min(len(lines), i + context_lines + 1)
                    
                    # Calculate relevance score
                    relevance_score = 0.0
                    
                    # Higher score for business logic indicators
                    for keyword in business_keywords:
                        if keyword in line_lower:
                            relevance_score += 0.3
                    
                    # Higher score for method definitions
                    if 'public' in line_lower and '(' in line and ')' in line:
                        relevance_score += 0.4
                    
                    # Higher score for multiple query terms
                    matching_terms = sum(1 for term in query_terms if term in line_lower)
                    relevance_score += matching_terms * 0.2
                    
                    # Higher score if in Action or Service class
                    if 'action' in str(java_file).lower() or 'service' in str(java_file).lower():
                        relevance_score += 0.3
                    
                    # Filter out low relevance results
                    if relevance_score > 0.1:
                        results.append({
                            "file_path": str(java_file.relative_to(repo_path_obj)),
                            "line_number": i + 1,
                            "matching_line": line.strip(),
                            "context": "".join(lines[start_line:end_line]),
                            "relevance_score": round(relevance_score, 2),
                            "file_type": "java",
                            "is_business_logic": any(keyword in line_lower for keyword in business_keywords)
                        })
                        
        except Exception as e:
            continue
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return results[:25]  # Return top 25 most relevant results


def create_codebase_agent(repo_path: str) -> Agent:
    """Create the main codebase analysis agent"""
    
    system_prompt = f"""You are a senior software architect and legacy modernization specialist with deep expertise in enterprise Java systems, particularly Struts-to-modern framework migrations.

ANALYSIS OBJECTIVES:
You help extract business logic from 200k+ LOC enterprise codebases for AI-assisted modernization. Your primary goal is creating clean business logic documentation that future AI can use to rewrite applications in modern frameworks.

CURRENT CODEBASE: {repo_path}

TECHNOLOGY STACK EXPERTISE:
- Legacy: Struts Actions, ActionForms, struts-config.xml, JSPs
- Modern: Spring Boot, Spring MVC, REST APIs, Angular
- Languages: Java, JavaScript/TypeScript, Perl
- Data: Oracle stored procedures, Snowflake, JDBC patterns
- Infrastructure: Kubernetes, AWS services

CORE RESPONSIBILITIES:
1. **Business Logic Extraction**: Identify and document core business rules separate from framework plumbing
2. **Data Flow Analysis**: Map how data moves from web → action → service → database
3. **Migration Assessment**: Categorize components by modernization complexity and risk
4. **Knowledge Generation**: Create structured documentation for future AI-assisted rewrites

ANALYSIS METHODOLOGY:
- **Struts Actions**: Focus on execute() methods, ignore framework setup code
- **Business Rules**: Extract validation, calculation, transformation logic
- **Dependencies**: Map service calls, DAO usage, external integrations  
- **Data Patterns**: Identify form handling, session management, database interactions
- **Risk Factors**: Assess tight coupling, complex business logic, external dependencies

OUTPUT REQUIREMENTS:
- Generate migration-ready documentation
- Create knowledge graphs showing business logic relationships
- Provide risk assessments (low/medium/high) without effort estimates
- Focus on business value preservation during modernization

Always separate business intent from implementation details. Your analysis enables confident modernization decisions."""

    user_prompt = """I need you to analyze this legacy enterprise codebase for modernization planning. 

PRIORITY TASKS:
1. Extract business logic from Struts Action classes (focus on execute() methods)
2. Map data flow patterns from web requests through to database operations  
3. Identify reusable business rules that must be preserved during migration
4. Generate documentation that will enable future AI-assisted code rewrites

Please start by analyzing the codebase structure and then proceed with business logic extraction. Focus on separating business intent from framework implementation details."""

    # Initialize Bedrock client
    bedrock_client = BedrockClient(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-west-2"
    )

    # Create the agent
    agent = Agent(
        client=bedrock_client,
        system_prompt=system_prompt,
        tools=[
            analyze_codebase_structure,
            analyze_struts_configuration,
            extract_business_logic,
            build_data_flow_map,
            generate_knowledge_graph,
            create_migration_documentation,
            search_business_logic
        ]
    )

    return agent


def main():
    """Main entry point for the codebase analysis agent"""
    
    try:
        # Check for required dependencies
        import strands_agents
        print("✓ Strands Agents loaded successfully")
    except ImportError:
        print("✗ Error: strands-agents not installed. Run: pip install strands-agents")
        return
    
    try:
        import javalang
        print("✓ Java parser loaded successfully")
    except ImportError:
        print("✗ Error: javalang not installed. Run: pip install javalang")
        return
    
    # Configuration
    repo_path = input("Enter the path to your codebase: ").strip()
    if not os.path.exists(repo_path):
        print(f"✗ Error: Path {repo_path} does not exist")
        return
    
    print(f"✓ Codebase path validated: {repo_path}")
    
    # Check AWS credentials
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            print("✗ Error: AWS credentials not configured. Set up AWS CLI or environment variables.")
            return
        print("✓ AWS credentials found")
    except Exception as e:
        print(f"✗ Error with AWS setup: {e}")
        return
    
    print(f"\n🚀 Initializing codebase analysis agent for: {repo_path}")
    
    try:
        # Create agent
        agent = create_codebase_agent(repo_path)
        print("✓ Agent created successfully")
    except Exception as e:
        print(f"✗ Error creating agent: {e}")
        return
    
    print("\nCodebase Analysis Agent ready!")
    print("Example queries:")
    print("- 'Analyze the overall structure and identify Struts patterns'")
    print("- 'Extract business logic from all Struts Action classes'") 
    print("- 'Analyze struts-config.xml to understand application mappings'")
    print("- 'Build a data flow map for the user authentication process'")
    print("- 'Search for payment processing business logic'")
    print("- 'Create migration documentation prioritized by risk level'")
    print("- 'Find all validation rules and business constraints'")
    print("\nType 'quit' to exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Run the agent
            response = agent.run(user_input)
            print(f"\nAgent Response:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Thanks for using the Codebase Analysis Agent!")


if __name__ == "__main__":
    main()
