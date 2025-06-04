# config.py
"""
Configuration for the Codebase Analysis Agent
Customize analysis patterns, AWS integration, and output formats
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for code analysis behavior"""
    
    # File patterns to analyze
    java_patterns: List[str] = None
    js_patterns: List[str] = None
    perl_patterns: List[str] = None
    
    # Framework detection patterns
    struts_indicators: List[str] = None
    spring_indicators: List[str] = None
    angular_indicators: List[str] = None
    
    # Business logic patterns
    business_method_patterns: List[str] = None
    exclude_patterns: List[str] = None
    
    # Risk assessment weights
    complexity_weight: float = 0.3
    dependency_weight: float = 0.4
    size_weight: float = 0.3
    
    def __post_init__(self):
        if self.java_patterns is None:
            self.java_patterns = ["*.java"]
        
        if self.js_patterns is None:
            self.js_patterns = ["*.js", "*.ts", "*.jsx", "*.tsx"]
        
        if self.perl_patterns is None:
            self.perl_patterns = ["*.pl", "*.pm", "*.pod"]
        
        if self.struts_indicators is None:
            self.struts_indicators = [
                "extends Action", "extends DispatchAction", 
                "ActionForm", "ActionForward", "ActionMapping"
            ]
        
        if self.spring_indicators is None:
            self.spring_indicators = [
                "@Controller", "@Service", "@Component", "@Repository",
                "@Autowired", "@Bean", "@Configuration"
            ]
        
        if self.angular_indicators is None:
            self.angular_indicators = [
                "@Component", "@Service", "@Injectable", 
                "import { Component }", "export class"
            ]
        
        if self.business_method_patterns is None:
            self.business_method_patterns = [
                "calculate", "process", "validate", "transform",
                "business", "rule", "logic", "execute", "perform"
            ]
        
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "test", "spec", "mock", "stub", "generated",
                "toString", "equals", "hashCode", "getter", "setter"
            ]


@dataclass 
class AWSConfig:
    """Configuration for AWS services integration"""
    
    region: str = "us-west-2"
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    
    # S3 configuration
    s3_bucket: str = None
    s3_prefix: str = "codebase-analysis"
    
    # Knowledge Base configuration
    knowledge_base_id: str = None
    knowledge_base_data_source_id: str = None
    
    # DynamoDB configuration
    dynamodb_table_name: str = None
    
    # CloudWatch configuration
    cloudwatch_log_group: str = "/aws/lambda/codebase-analysis-agent"
    
    def __post_init__(self):
        # Load from environment variables
        self.region = os.getenv("AWS_REGION", self.region)
        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", self.bedrock_model_id)
        self.s3_bucket = os.getenv("S3_BUCKET_NAME", self.s3_bucket)
        self.knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID", self.knowledge_base_id)
        self.dynamodb_table_name = os.getenv("DYNAMODB_TABLE_NAME", self.dynamodb_table_name)


@dataclass
class OutputConfig:
    """Configuration for analysis output formats and destinations"""
    
    # Output directories
    output_dir: str = "./analysis_output"
    temp_dir: str = "/tmp/codebase_analysis"
    
    # Output formats
    generate_json: bool = True
    generate_markdown: bool = True
    generate_html: bool = False
    generate_graphs: bool = True
    
    # Graph configuration
    graph_format: str = "png"  # png, svg, pdf
    graph_layout: str = "spring"  # spring, circular, hierarchical
    
    # Documentation settings
    include_code_snippets: bool = True
    max_snippet_lines: int = 20
    include_risk_details: bool = True
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)


# Default configurations
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
DEFAULT_AWS_CONFIG = AWSConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()


# Enhanced agent configuration with AWS integration
def create_enhanced_agent_config(
    analysis_config: AnalysisConfig = None,
    aws_config: AWSConfig = None,
    output_config: OutputConfig = None
) -> Dict[str, Any]:
    """Create enhanced agent configuration with AWS services"""
    
    if analysis_config is None:
        analysis_config = DEFAULT_ANALYSIS_CONFIG
    if aws_config is None:
        aws_config = DEFAULT_AWS_CONFIG
    if output_config is None:
        output_config = DEFAULT_OUTPUT_CONFIG
    
    return {
        "analysis": analysis_config,
        "aws": aws_config,
        "output": output_config,
        
        # Agent behavior configuration
        "agent_config": {
            "max_iterations": 50,
            "timeout_seconds": 300,
            "memory_limit_mb": 2048,
            "enable_caching": True,
            "cache_duration_hours": 24,
        },
        
        # Advanced prompts for different analysis types
        "prompts": {
            "struts_migration": """
            You are analyzing a Struts-based application for migration to modern frameworks. Focus on:
            
            BUSINESS LOGIC EXTRACTION:
            1. Identify Action classes - look for classes extending Action, DispatchAction, LookupDispatchAction
            2. Extract execute() and perform() method implementations - ignore framework setup
            3. Map ActionForm objects and their validation rules to business data models
            4. Analyze struts-config.xml action mappings, form-beans, and forwards
            5. Document business validation patterns separate from Struts validation framework
            6. Identify data transformation logic between forms and business objects
            7. Extract error handling patterns that represent business rules vs technical errors
            
            DATA FLOW ANALYSIS:
            1. Trace request flow: JSP → Action → Business Service → DAO → Database
            2. Map form data binding and transformation patterns
            3. Identify session management and state handling for business processes
            4. Document integration points with external services or systems
            
            MODERNIZATION FOCUS:
            1. Separate business intent from Struts-specific implementation
            2. Identify reusable business components vs framework-tied code
            3. Document business rules that must be preserved during migration
            4. Flag complex business logic that requires careful testing during migration
            """,
            
            "business_logic_extraction": """
            Extract core business logic while filtering out infrastructure and framework code:
            
            BUSINESS LOGIC INDICATORS:
            1. Methods containing calculations, transformations, or business rules
            2. Validation logic that enforces business constraints (not just data format)
            3. Workflow and process orchestration logic
            4. Business decision points and conditional logic
            5. Data aggregation and reporting calculations
            6. Integration logic with business meaning (not just technical connectivity)
            
            IGNORE THESE PATTERNS:
            1. Getters/setters and basic POJO methods
            2. Framework initialization and configuration code
            3. Technical validation (null checks, format validation)
            4. Database connection and transaction management
            5. Logging, monitoring, and debugging code
            6. UI presentation logic and formatting
            
            DOCUMENTATION REQUIREMENTS:
            1. Describe business purpose in plain language
            2. Identify inputs, outputs, and business rules
            3. Map dependencies between business components
            4. Flag complex business logic requiring domain expertise
            """,
            
            "modernization_assessment": """
            Assess modernization complexity and create actionable migration strategy:
            
            COMPLEXITY ASSESSMENT:
            1. LOW RISK: Simple CRUD operations, basic validations, standard patterns
            2. MEDIUM RISK: Complex business logic, multiple dependencies, custom patterns  
            3. HIGH RISK: Tightly coupled components, complex state management, legacy integrations
            
            MIGRATION STRATEGY:
            1. Identify components that can be migrated independently (loose coupling)
            2. Suggest modern equivalents: Struts Action → Spring Controller, ActionForm → DTO/Entity
            3. Propose incremental migration paths to minimize business disruption
            4. Document integration points that need special handling during migration
            5. Identify business-critical paths requiring extensive testing
            
            MODERNIZATION RECOMMENDATIONS:
            1. Suggest API-first approaches for complex business services
            2. Recommend separation of concerns improvements
            3. Identify opportunities for microservices extraction
            4. Flag components suitable for automated conversion vs manual rewrite
            
            RISK FACTORS:
            1. Business logic embedded in presentation layer
            2. Complex interdependencies between business components  
            3. Custom framework extensions or modifications
            4. Integration with legacy systems or databases
            5. Complex business rules requiring domain expertise
            """
        }
    }


# Deployment configurations for different environments

LAMBDA_DEPLOYMENT_CONFIG = {
    "runtime": "python3.9",
    "memory": 3008,
    "timeout": 900,
    "environment_variables": {
        "PYTHONPATH": "/var/task",
        "AWS_REGION": "${AWS::Region}",
    },
    "layers": [
        "arn:aws:lambda:us-west-2:123456789012:layer:strands-agents:1"
    ]
}

FARGATE_DEPLOYMENT_CONFIG = {
    "cpu": 2048,
    "memory": 4096,
    "task_definition": {
        "family": "codebase-analysis-agent",
        "network_mode": "awsvpc",
        "requires_compatibilities": ["FARGATE"],
        "execution_role_arn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
        "task_role_arn": "arn:aws:iam::123456789012:role/codebase-analysis-task-role"
    },
    "service": {
        "desired_count": 1,
        "launch_type": "FARGATE",
        "platform_version": "LATEST"
    }
}

EC2_DEPLOYMENT_CONFIG = {
    "instance_type": "c5.2xlarge",
    "ami_id": "ami-0abcdef1234567890",  # Amazon Linux 2
    "security_groups": ["sg-codebase-analysis"],
    "user_data": """#!/bin/bash
yum update -y
yum install -y python3 git
pip3 install strands-agents boto3
# Additional setup commands
"""
}


# CloudFormation template for AWS infrastructure
CLOUDFORMATION_TEMPLATE = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "Infrastructure for Codebase Analysis Agent",
    
    "Parameters": {
        "BucketName": {
            "Type": "String",
            "Description": "S3 bucket for storing analysis results"
        },
        "Environment": {
            "Type": "String",
            "Default": "dev",
            "AllowedValues": ["dev", "staging", "prod"]
        }
    },
    
    "Resources": {
        "AnalysisResultsBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": {"Ref": "BucketName"},
                "VersioningConfiguration": {"Status": "Enabled"},
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True
                }
            }
        },
        
        "CodebaseMetadataTable": {
            "Type": "AWS::DynamoDB::Table",
            "Properties": {
                "TableName": {"Fn::Sub": "codebase-metadata-${Environment}"},
                "BillingMode": "PAY_PER_REQUEST",
                "AttributeDefinitions": [
                    {"AttributeName": "repo_id", "AttributeType": "S"},
                    {"AttributeName": "entity_id", "AttributeType": "S"}
                ],
                "KeySchema": [
                    {"AttributeName": "repo_id", "KeyType": "HASH"},
                    {"AttributeName": "entity_id", "KeyType": "RANGE"}
                ]
            }
        },
        
        "KnowledgeBase": {
            "Type": "AWS::Bedrock::KnowledgeBase",
            "Properties": {
                "Name": {"Fn::Sub": "codebase-knowledge-base-${Environment}"},
                "Description": "Knowledge base for codebase analysis and search",
                "RoleArn": {"Fn::GetAtt": "KnowledgeBaseRole.Arn"},
                "KnowledgeBaseConfiguration": {
                    "Type": "VECTOR",
                    "VectorKnowledgeBaseConfiguration": {
                        "EmbeddingModelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v1"
                    }
                },
                "StorageConfiguration": {
                    "Type": "OPENSEARCH_SERVERLESS",
                    "OpensearchServerlessConfiguration": {
                        "CollectionArn": {"Fn::GetAtt": "OpenSearchCollection.Arn"},
                        "VectorIndexName": "codebase-index",
                        "FieldMapping": {
                            "VectorField": "embedding",
                            "TextField": "content",
                            "MetadataField": "metadata"
                        }
                    }
                }
            }
        },
        
        "AgentExecutionRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                },
                "ManagedPolicyArns": [
                    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                ],
                "Policies": [{
                    "PolicyName": "CodebaseAnalysisPermissions",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "bedrock:InvokeModel",
                                    "bedrock:InvokeModelWithResponseStream"
                                ],
                                "Resource": "arn:aws:bedrock:*:*:model/anthropic.claude-3-5-sonnet-*"
                            },
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "s3:GetObject",
                                    "s3:PutObject",
                                    "s3:DeleteObject"
                                ],
                                "Resource": {"Fn::Sub": "${AnalysisResultsBucket}/*"}
                            },
                            {
                                "Effect": "Allow",
                                "Action": [
                                    "dynamodb:GetItem",
                                    "dynamodb:PutItem",
                                    "dynamodb:UpdateItem",
                                    "dynamodb:DeleteItem",
                                    "dynamodb:Query",
                                    "dynamodb:Scan"
                                ],
                                "Resource": {"Fn::GetAtt": "CodebaseMetadataTable.Arn"}
                            }
                        ]
                    }
                }]
            }
        }
    },
    
    "Outputs": {
        "BucketName": {
            "Description": "Name of the S3 bucket for analysis results",
            "Value": {"Ref": "AnalysisResultsBucket"}
        },
        "TableName": {
            "Description": "Name of the DynamoDB table for metadata",
            "Value": {"Ref": "CodebaseMetadataTable"}
        },
        "KnowledgeBaseId": {
            "Description": "ID of the Bedrock Knowledge Base",
            "Value": {"Ref": "KnowledgeBase"}
        }
    }
}


# Usage examples and integration patterns

def get_struts_migration_config():
    """Get configuration optimized for Struts to modern framework migration"""
    config = create_enhanced_agent_config()
    
    # Enhance for Struts migration
    config["analysis"].business_method_patterns.extend([
        "execute", "perform", "doGet", "doPost", "validate"
    ])
    
    config["analysis"].struts_indicators.extend([
        "struts-config.xml", "validation.xml", "MessageResources"
    ])
    
    return config


def get_enterprise_scale_config():
    """Get configuration for very large enterprise codebases"""
    config = create_enhanced_agent_config()
    
    # Optimize for large scale
    config["agent_config"]["max_iterations"] = 100
    config["agent_config"]["timeout_seconds"] = 1800
    config["agent_config"]["memory_limit_mb"] = 8192
    config["agent_config"]["enable_parallel_processing"] = True
    config["agent_config"]["chunk_size_files"] = 1000
    
    return config


if __name__ == "__main__":
    # Example: Generate configuration files
    import json
    
    # Create different configuration profiles
    configs = {
        "struts_migration": get_struts_migration_config(),
        "enterprise_scale": get_enterprise_scale_config(),
        "default": create_enhanced_agent_config()
    }
    
    # Save configurations
    for name, config in configs.items():
        with open(f"config_{name}.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    # Save CloudFormation template
    with open("infrastructure.yaml", "w") as f:
        import yaml
        yaml.dump(CLOUDFORMATION_TEMPLATE, f, default_flow_style=False)
    
    print("Configuration files generated successfully!")
