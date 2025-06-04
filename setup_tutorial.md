# 🚀 Complete Local Setup Guide for Codebase Analysis Agent

This guide will walk you through setting up the AWS Strands-based codebase analysis agent on your local machine.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for large codebases)
- **Storage**: 2GB free space for dependencies and analysis outputs

### Required Accounts & Access
- **AWS Account** with Bedrock access in `us-west-2` region
- **GitHub Account** (for downloading dependencies)
- **Admin/sudo privileges** on your local machine

---

## 🛠️ Step 1: Install Python and Dependencies

### Windows Setup

1. **Install Python 3.8+**
   ```powershell
   # Download from python.org or use chocolatey
   choco install python --version=3.11.0
   
   # Verify installation
   python --version
   pip --version
   ```

2. **Install Git**
   ```powershell
   choco install git
   ```

3. **Install Visual Studio Build Tools** (required for some Python packages)
   ```powershell
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   # Or use chocolatey
   choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
   ```

### macOS Setup

1. **Install Python 3.8+**
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.11
   
   # Or download from python.org
   # Verify installation
   python3 --version
   pip3 --version
   ```

2. **Install Git**
   ```bash
   brew install git
   ```

3. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

### Linux (Ubuntu/Debian) Setup

1. **Update system and install Python**
   ```bash
   sudo apt update
   sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev
   sudo apt install -y git build-essential
   
   # Create symbolic links
   sudo ln -s /usr/bin/python3.11 /usr/bin/python
   sudo ln -s /usr/bin/pip3 /usr/bin/pip
   
   # Verify installation
   python --version
   pip --version
   ```

---

## ☁️ Step 2: Configure AWS Access

### Install AWS CLI

**Windows:**
```powershell
# Using installer
# Download from: https://aws.amazon.com/cli/
# Or use chocolatey
choco install awscli
```

**macOS:**
```bash
brew install awscli
```

**Linux:**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configure AWS Credentials

1. **Get your AWS credentials** from your AWS console:
   - Access Key ID
   - Secret Access Key
   - Region: `us-west-2` (required for Bedrock)

2. **Configure AWS CLI**
   ```bash
   aws configure
   ```
   
   Enter when prompted:
   ```
   AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
   AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
   Default region name [None]: us-west-2
   Default output format [None]: json
   ```

3. **Test AWS connection**
   ```bash
   aws sts get-caller-identity
   ```
   
   You should see your AWS account details.

### Enable Bedrock Model Access

1. **Open AWS Console** → Bedrock → Model access
2. **Request access** to Claude 3.5 Sonnet models
   - Go to Model access in the Bedrock console
   - Click "Manage model access"
   - Enable "Claude 3.5 Sonnet" by Anthropic
   - Submit the request

> ⚠️ **Important**: Model access approval can take a few minutes to several hours

3. **Verify model access**
   ```bash
   aws bedrock list-foundation-models --region us-west-2 --query 'modelSummaries[?contains(modelId, `claude-3-5-sonnet`)]'
   ```

---

## 📁 Step 3: Project Setup

### Create Project Directory

```bash
# Create and navigate to project directory
mkdir codebase-analysis-agent
cd codebase-analysis-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify virtual environment
which python  # Should show path in your venv folder
```

### Download Project Files

**Option A: Clone from repository (if available)**
```bash
git clone YOUR_REPOSITORY_URL .
```

**Option B: Create files manually**

Create the project structure:
```bash
mkdir analysis_output
mkdir temp
touch codebase_agent.py
touch config.py  
touch requirements.txt
touch .env
touch README.md
```

### Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Core dependencies
strands-agents>=0.1.0
boto3>=1.34.0
botocore>=1.34.0

# Code analysis
javalang>=0.13.0
tree-sitter>=0.20.0

# Data processing
networkx>=3.1
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
graphviz>=0.20.0

# Utilities
pathlib2>=2.3.7
python-dotenv>=1.0.0
click>=8.1.0
tqdm>=4.65.0

# Development
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
EOF
```

### Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify key installations
python -c "import strands_agents; print('✓ Strands Agents installed')"
python -c "import javalang; print('✓ Java parser installed')"  
python -c "import boto3; print('✓ AWS SDK installed')"
```

---

## ⚙️ Step 4: Configuration

### Create Environment File

Create `.env` file with your configuration:

```bash
cat > .env << 'EOF'
# AWS Configuration
AWS_REGION=us-west-2
AWS_PROFILE=default

# Bedrock Configuration  
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Optional: S3 bucket for storing analysis results
# S3_BUCKET_NAME=your-codebase-analysis-bucket

# Optional: Knowledge Base ID if using Bedrock Knowledge Bases
# KNOWLEDGE_BASE_ID=your_knowledge_base_id

# Analysis Configuration
MAX_FILES_TO_ANALYZE=500
CONTEXT_LINES=5
OUTPUT_DIRECTORY=./analysis_output

# Performance Tuning
ENABLE_CACHING=true
CACHE_DURATION_HOURS=24
MAX_MEMORY_MB=4096
EOF
```

### Copy Agent Code

Copy the main agent code from the artifacts into `codebase_agent.py` and `config.py`.

### Test Configuration

Create a simple test script:

```bash
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify local setup
"""
import os
import sys
from pathlib import Path

def test_python_version():
    """Test Python version"""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version.split()[0]} installed")
    return True

def test_dependencies():
    """Test required dependencies"""
    try:
        import strands_agents
        print("✓ strands-agents imported successfully")
    except ImportError as e:
        print(f"✗ strands-agents import failed: {e}")
        return False
    
    try:
        import javalang
        print("✓ javalang imported successfully")  
    except ImportError as e:
        print(f"✗ javalang import failed: {e}")
        return False
        
    try:
        import boto3
        print("✓ boto3 imported successfully")
    except ImportError as e:
        print(f"✗ boto3 import failed: {e}")
        return False
        
    return True

def test_aws_credentials():
    """Test AWS credentials and Bedrock access"""
    try:
        import boto3
        
        # Test basic AWS access
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            print("✗ AWS credentials not found")
            return False
            
        print("✓ AWS credentials configured")
        
        # Test Bedrock access
        bedrock = boto3.client('bedrock', region_name='us-west-2')
        models = bedrock.list_foundation_models()
        
        claude_models = [m for m in models['modelSummaries'] 
                        if 'claude-3-5-sonnet' in m['modelId']]
        
        if not claude_models:
            print("✗ Claude 3.5 Sonnet not accessible in Bedrock")
            print("  Please enable model access in AWS Bedrock console")
            return False
            
        print("✓ Bedrock and Claude 3.5 Sonnet accessible")
        return True
        
    except Exception as e:
        print(f"✗ AWS/Bedrock test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    if not os.path.exists('.env'):
        print("✗ .env file not found")
        return False
    print("✓ .env file exists")
    
    # Create output directory
    output_dir = Path('./analysis_output')
    output_dir.mkdir(exist_ok=True)
    print("✓ Output directory created")
    
    return True

def main():
    """Run all tests"""
    print("🔍 Testing Codebase Analysis Agent Setup\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies), 
        ("Environment", test_environment),
        ("AWS & Bedrock", test_aws_credentials),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)
    
    print(f"\n{'='*50}")
    if all(results):
        print("🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Place your codebase in a local directory") 
        print("2. Run: python codebase_agent.py")
        print("3. Enter the path to your codebase when prompted")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run the test
python test_setup.py
```

---

## 🧪 Step 5: Verify Setup with Sample Run

### Create Test Java Project

Create a simple test codebase to verify the agent works:

```bash
mkdir test_codebase
cd test_codebase

# Create a simple Struts Action for testing
mkdir -p src/main/java/com/example/actions
cat > src/main/java/com/example/actions/UserAction.java << 'EOF'
package com.example.actions;

import org.apache.struts.action.Action;
import org.apache.struts.action.ActionForm;
import org.apache.struts.action.ActionMapping;
import org.apache.struts.action.ActionForward;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class UserAction extends Action {
    
    public ActionForward execute(ActionMapping mapping, ActionForm form,
                                HttpServletRequest request, HttpServletResponse response) {
        
        // Business logic: User validation
        String username = request.getParameter("username");
        if (username == null || username.trim().isEmpty()) {
            return mapping.findForward("input");
        }
        
        // Business logic: Calculate user score
        int score = calculateUserScore(username);
        request.setAttribute("userScore", score);
        
        // Business logic: Determine user level
        String userLevel = determineUserLevel(score);
        request.setAttribute("userLevel", userLevel);
        
        return mapping.findForward("success");
    }
    
    private int calculateUserScore(String username) {
        // Business rule: Score based on username length and content
        int score = username.length() * 10;
        if (username.contains("admin")) {
            score += 100;
        }
        return score;
    }
    
    private String determineUserLevel(int score) {
        // Business rule: User level determination
        if (score > 150) return "Premium";
        if (score > 50) return "Standard";
        return "Basic";
    }
}
EOF

# Create struts-config.xml
mkdir -p src/main/webapp/WEB-INF
cat > src/main/webapp/WEB-INF/struts-config.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE struts-config PUBLIC
    "-//Apache Software Foundation//DTD Struts Configuration 1.3//EN"
    "http://struts.apache.org/dtds/struts-config_1_3.dtd">

<struts-config>
    <form-beans>
        <form-bean name="userForm" type="com.example.forms.UserForm"/>
    </form-beans>
    
    <action-mappings>
        <action path="/user" 
                type="com.example.actions.UserAction"
                name="userForm"
                scope="request">
            <forward name="success" path="/user-success.jsp"/>
            <forward name="input" path="/user-input.jsp"/>
        </action>
    </action-mappings>
</struts-config>
EOF

cd ..
```

### Run First Analysis

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the agent
python codebase_agent.py
```

When prompted, enter: `./test_codebase`

Try these test queries:
1. `"Analyze the overall structure and identify Struts patterns"`
2. `"Extract business logic from all Action classes"`
3. `"Search for user validation business logic"`

---

## 🐛 Troubleshooting Common Issues

### Python/Dependency Issues

**Issue**: `ModuleNotFoundError: No module named 'strands_agents'`
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall in virtual environment
pip install strands-agents
```

**Issue**: `Building wheel for javalang failed`
```bash
# Solution: Install build tools
# Windows: Install Visual Studio Build Tools
# macOS: xcode-select --install  
# Linux: sudo apt install build-essential python3-dev
```

### AWS Issues

**Issue**: `NoCredentialsError: Unable to locate credentials`
```bash
# Solution: Configure AWS credentials
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

**Issue**: `AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel`
```bash
# Solution: Add Bedrock permissions to your IAM user/role
# Required policy:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:*:*:model/anthropic.claude-3-5-sonnet-*"
        }
    ]
}
```

**Issue**: `ValidationException: The provided model identifier is invalid`
```bash
# Solution: Enable model access in Bedrock console
# 1. Go to AWS Bedrock console
# 2. Navigate to "Model access"  
# 3. Click "Manage model access"
# 4. Enable Claude 3.5 Sonnet models
# 5. Wait for approval (can take up to 24 hours)
```

### Performance Issues

**Issue**: Agent runs slowly on large codebases
```bash
# Solution: Adjust configuration in .env
MAX_FILES_TO_ANALYZE=100  # Reduce for initial testing
ENABLE_CACHING=true       # Enable caching
```

**Issue**: Out of memory errors
```bash
# Solution: Increase system limits or process in chunks
MAX_MEMORY_MB=8192        # Increase if you have more RAM
# Or analyze smaller portions of the codebase
```

### Code Analysis Issues

**Issue**: `UnicodeDecodeError` when reading files
```bash
# Solution: The agent handles this automatically, but you can:
# 1. Check file encodings in your codebase
# 2. Convert files to UTF-8 if needed
find . -name "*.java" -exec file {} \; | grep -v UTF-8
```

**Issue**: Java parsing errors
```bash
# Solution: 
# 1. Ensure Java files are syntactically valid
# 2. Check Java version compatibility (agent supports Java 8+)
# 3. Review error messages for specific parsing issues
```

---

## 🎯 Next Steps

Once setup is complete:

1. **Run on your actual codebase**
   ```bash
   python codebase_agent.py
   # Enter path to your Struts application
   ```

2. **Try advanced queries**
   - `"Create migration documentation for all high-risk Struts components"`
   - `"Build a knowledge graph of business logic relationships"`
   - `"Analyze data flow for the order processing system"`

3. **Customize for your domain**
   - Edit `config.py` to add your business-specific patterns
   - Modify prompts for your industry/domain
   - Add custom tools for your specific analysis needs

4. **Set up CI/CD integration** (see next section)

---

## 🔄 CI/CD Integration

### Containerization with Podman

Create a Podmanfile (equivalent to Dockerfile):

```bash
cat > Containerfile << 'EOF'
FROM registry.access.redhat.com/ubi9/ubi:latest

# Install Python and system dependencies
RUN dnf update -y && \
    dnf install -y python3.11 python3.11-pip python3.11-devel \
                   git gcc gcc-c++ make \
                   java-11-openjdk-devel && \
    dnf clean all

# Set Python alias
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY codebase_agent.py config.py ./
COPY analysis_scripts/ ./analysis_scripts/

# Create directories for analysis output
RUN mkdir -p /app/analysis_output /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -s /bin/bash analyst && \
    chown -R analyst:analyst /app
USER analyst

# Default command
CMD ["python", "codebase_agent.py", "--batch-mode"]
EOF
```

Build and test the container:

```bash
# Build the container
podman build -t codebase-analysis-agent .

# Test the container locally
podman run -it --rm \
  -v $(pwd)/test_codebase:/workspace:ro \
  -v $(pwd)/analysis_output:/app/analysis_output \
  -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
  -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
  -e AWS_DEFAULT_REGION=us-west-2 \
  codebase-analysis-agent
```

### Batch Mode Script

Create a batch mode for automated analysis:

```bash
cat > analysis_scripts/batch_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Batch analysis script for CI/CD integration
Analyzes codebase and generates reports without interactive prompts
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from codebase_agent import create_codebase_agent, CodebaseAnalyzer

def run_batch_analysis(repo_path: str, output_dir: str, config: dict = None):
    """Run automated codebase analysis"""
    
    print(f"🚀 Starting batch analysis of {repo_path}")
    start_time = datetime.now()
    
    # Create agent
    agent = create_codebase_agent(repo_path)
    
    # Define analysis tasks
    analysis_tasks = [
        {
            "name": "structure_analysis",
            "query": "Analyze the overall codebase structure and identify all Struts patterns and frameworks used",
            "output_file": "structure_analysis.json"
        },
        {
            "name": "business_logic_extraction", 
            "query": "Extract all business logic from Struts Action classes and Spring components, focusing on core business rules",
            "output_file": "business_logic.json"
        },
        {
            "name": "struts_configuration",
            "query": "Analyze all Struts configuration files to understand application mappings and flows",
            "output_file": "struts_config_analysis.json"
        },
        {
            "name": "data_flow_analysis",
            "query": "Build comprehensive data flow maps showing how data moves through the entire system",
            "output_file": "data_flow.json"
        },
        {
            "name": "migration_documentation",
            "query": "Create detailed migration documentation with risk assessment for all components prioritized by complexity",
            "output_file": "migration_plan.md"
        },
        {
            "name": "knowledge_graph",
            "query": "Generate a complete knowledge graph of business logic relationships and dependencies",
            "output_file": "knowledge_graph.json"
        }
    ]
    
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for task in analysis_tasks:
        print(f"\n📊 Running {task['name']}...")
        try:
            response = agent.run(task['query'])
            
            # Save response to file
            output_file = output_path / task['output_file']
            if task['output_file'].endswith('.json'):
                # Try to parse as JSON, fallback to text
                try:
                    json_data = json.loads(response) if isinstance(response, str) else response
                    with open(output_file, 'w') as f:
                        json.dump(json_data, f, indent=2)
                except (json.JSONDecodeError, TypeError):
                    with open(output_file, 'w') as f:
                        f.write(str(response))
            else:
                with open(output_file, 'w') as f:
                    f.write(str(response))
            
            results[task['name']] = {
                "status": "success",
                "output_file": str(output_file),
                "response_length": len(str(response))
            }
            print(f"✅ {task['name']} completed → {output_file}")
            
        except Exception as e:
            print(f"❌ {task['name']} failed: {e}")
            results[task['name']] = {
                "status": "failed", 
                "error": str(e)
            }
    
    # Generate summary report
    end_time = datetime.now()
    duration = end_time - start_time
    
    summary = {
        "analysis_metadata": {
            "repository_path": repo_path,
            "analysis_date": start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "total_tasks": len(analysis_tasks),
            "successful_tasks": len([r for r in results.values() if r["status"] == "success"]),
            "failed_tasks": len([r for r in results.values() if r["status"] == "failed"])
        },
        "task_results": results,
        "output_directory": output_dir
    }
    
    # Save summary
    summary_file = output_path / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Batch analysis completed in {duration}")
    print(f"📋 Summary saved to {summary_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Batch codebase analysis for CI/CD")
    parser.add_argument("--repo-path", required=True, help="Path to codebase to analyze")
    parser.add_argument("--output-dir", default="./analysis_output", help="Output directory for results")
    parser.add_argument("--config-file", help="Optional configuration file")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.repo_path):
        print(f"❌ Repository path does not exist: {args.repo_path}")
        sys.exit(1)
    
    # Load config if provided
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Run analysis
    try:
        summary = run_batch_analysis(args.repo_path, args.output_dir, config)
        
        # Exit with appropriate code
        if summary["analysis_metadata"]["failed_tasks"] > 0:
            print(f"⚠️  Analysis completed with {summary['analysis_metadata']['failed_tasks']} failures")
            sys.exit(1)
        else:
            print("✅ All analysis tasks completed successfully")
            sys.exit(0)
            
    except Exception as e:
        print(f"💥 Batch analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x analysis_scripts/batch_analysis.py
```

### GitLab CI/CD Pipeline

Create `.gitlab-ci.yml`:

```yaml
cat > .gitlab-ci.yml << 'EOF'
stages:
  - build
  - analyze
  - report
  - deploy

variables:
  PODMAN_DRIVER: overlay2
  PODMAN_TLS_CERTDIR: ""
  ANALYSIS_OUTPUT_DIR: "analysis_results"

# Build the analysis container
build_analyzer:
  stage: build
  image: quay.io/podman/stable
  services:
    - name: quay.io/podman/stable:latest
      alias: podman
  before_script:
    - podman info
  script:
    - podman build -t codebase-analysis-agent:$CI_COMMIT_SHORT_SHA .
    - podman save codebase-analysis-agent:$CI_COMMIT_SHORT_SHA | gzip > analysis-agent.tar.gz
  artifacts:
    paths:
      - analysis-agent.tar.gz
    expire_in: 1 hour
  only:
    changes:
      - Containerfile
      - requirements.txt
      - codebase_agent.py
      - config.py
      - analysis_scripts/*

# Analyze the main codebase
analyze_codebase:
  stage: analyze
  image: quay.io/podman/stable
  services:
    - name: quay.io/podman/stable:latest
      alias: podman
  dependencies:
    - build_analyzer
  before_script:
    - podman load < analysis-agent.tar.gz
  script:
    # Run the analysis on the current repository
    - mkdir -p $ANALYSIS_OUTPUT_DIR
    - |
      podman run --rm \
        -v $CI_PROJECT_DIR:/workspace:ro \
        -v $CI_PROJECT_DIR/$ANALYSIS_OUTPUT_DIR:/app/analysis_output \
        -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
        -e AWS_DEFAULT_REGION=us-west-2 \
        codebase-analysis-agent:$CI_COMMIT_SHORT_SHA \
        python analysis_scripts/batch_analysis.py \
        --repo-path /workspace \
        --output-dir /app/analysis_output
  artifacts:
    paths:
      - $ANALYSIS_OUTPUT_DIR/
    expire_in: 30 days
    reports:
      junit: $ANALYSIS_OUTPUT_DIR/analysis_summary.json
  only:
    - main
    - develop
    - merge_requests

# Generate migration reports
generate_reports:
  stage: report
  image: python:3.11-slim
  dependencies:
    - analyze_codebase
  before_script:
    - pip install jinja2 matplotlib pandas
  script:
    - python analysis_scripts/generate_reports.py --input-dir $ANALYSIS_OUTPUT_DIR
  artifacts:
    paths:
      - $ANALYSIS_OUTPUT_DIR/reports/
    expire_in: 90 days
  only:
    - main

# Deploy results to S3 (optional)
deploy_results:
  stage: deploy
  image: amazon/aws-cli:latest
  dependencies:
    - generate_reports
  script:
    - |
      if [ ! -z "$S3_RESULTS_BUCKET" ]; then
        aws s3 sync $ANALYSIS_OUTPUT_DIR/ s3://$S3_RESULTS_BUCKET/analysis-results/$CI_COMMIT_SHORT_SHA/ \
          --exclude "*.git/*" \
          --include "*.json" \
          --include "*.md" \
          --include "*.html"
        echo "Results deployed to s3://$S3_RESULTS_BUCKET/analysis-results/$CI_COMMIT_SHORT_SHA/"
      fi
  only:
    - main
  when: manual

# Incremental analysis for MRs
analyze_changes:
  stage: analyze
  image: quay.io/podman/stable
  services:
    - name: quay.io/podman/stable:latest
      alias: podman
  dependencies:
    - build_analyzer
  before_script:
    - podman load < analysis-agent.tar.gz
    - apk add --no-cache git
  script:
    # Get list of changed files
    - git diff --name-only $CI_MERGE_REQUEST_TARGET_BRANCH_NAME..HEAD > changed_files.txt
    - |
      if grep -q "\.java$\|\.xml$\|\.js$\|\.pl$" changed_files.txt; then
        echo "Code changes detected, running incremental analysis..."
        mkdir -p $ANALYSIS_OUTPUT_DIR
        podman run --rm \
          -v $CI_PROJECT_DIR:/workspace:ro \
          -v $CI_PROJECT_DIR/$ANALYSIS_OUTPUT_DIR:/app/analysis_output \
          -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
          -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
          -e AWS_DEFAULT_REGION=us-west-2 \
          codebase-analysis-agent:$CI_COMMIT_SHORT_SHA \
          python analysis_scripts/incremental_analysis.py \
          --repo-path /workspace \
          --changed-files /workspace/changed_files.txt \
          --output-dir /app/analysis_output
      else
        echo "No code changes detected, skipping analysis"
      fi
  artifacts:
    paths:
      - $ANALYSIS_OUTPUT_DIR/
    expire_in: 7 days
  only:
    - merge_requests
EOF
```

### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
cat > Jenkinsfile << 'EOF'
pipeline {
    agent any
    
    environment {
        AWS_DEFAULT_REGION = 'us-west-2'
        ANALYSIS_OUTPUT_DIR = 'analysis_results'
        PODMAN_IMAGE = 'codebase-analysis-agent'
    }
    
    stages {
        stage('Build Analysis Container') {
            when {
                anyOf {
                    changeset "Containerfile"
                    changeset "requirements.txt"
                    changeset "codebase_agent.py"
                    changeset "analysis_scripts/*"
                }
            }
            steps {
                script {
                    sh '''
                        podman build -t ${PODMAN_IMAGE}:${BUILD_NUMBER} .
                        podman tag ${PODMAN_IMAGE}:${BUILD_NUMBER} ${PODMAN_IMAGE}:latest
                    '''
                }
            }
        }
        
        stage('Codebase Analysis') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    changeRequest()
                }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        mkdir -p ${ANALYSIS_OUTPUT_DIR}
                        
                        podman run --rm \
                            -v ${WORKSPACE}:/workspace:ro \
                            -v ${WORKSPACE}/${ANALYSIS_OUTPUT_DIR}:/app/analysis_output \
                            -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
                            -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
                            -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
                            ${PODMAN_IMAGE}:latest \
                            python analysis_scripts/batch_analysis.py \
                            --repo-path /workspace \
                            --output-dir /app/analysis_output
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: "${ANALYSIS_OUTPUT_DIR}/**/*", fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: "${ANALYSIS_OUTPUT_DIR}",
                        reportFiles: '*.html',
                        reportName: 'Codebase Analysis Report'
                    ])
                }
            }
        }
        
        stage('Migration Risk Assessment') {
            when { branch 'main' }
            steps {
                script {
                    sh '''
                        python3 analysis_scripts/risk_assessment.py \
                            --input-dir ${ANALYSIS_OUTPUT_DIR} \
                            --output-file ${ANALYSIS_OUTPUT_DIR}/risk_report.json
                    '''
                    
                    // Parse risk assessment and fail if high-risk changes detected
                    def riskReport = readJSON file: "${ANALYSIS_OUTPUT_DIR}/risk_report.json"
                    if (riskReport.high_risk_components?.size() > 0) {
                        echo "⚠️ High-risk components detected: ${riskReport.high_risk_components}"
                        currentBuild.result = 'UNSTABLE'
                    }
                }
            }
        }
        
        stage('Deploy Results') {
            when { 
                branch 'main'
                environment name: 'DEPLOY_RESULTS', value: 'true'
            }
            steps {
                withCredentials([
                    string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        if [ ! -z "${S3_RESULTS_BUCKET}" ]; then
                            aws s3 sync ${ANALYSIS_OUTPUT_DIR}/ \
                                s3://${S3_RESULTS_BUCKET}/builds/${BUILD_NUMBER}/ \
                                --delete
                            echo "Results deployed to S3"
                        fi
                    '''
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            emailext (
                subject: "Codebase Analysis Completed - Build ${BUILD_NUMBER}",
                body: """
                Codebase analysis completed successfully.
                
                Build: ${BUILD_NUMBER}
                Branch: ${BRANCH_NAME}
                Results: ${BUILD_URL}artifact/${ANALYSIS_OUTPUT_DIR}/
                
                Check the analysis results for migration planning insights.
                """,
                to: "${env.CHANGE_AUTHOR_EMAIL ?: 'dev-team@company.com'}"
            )
        }
        failure {
            emailext (
                subject: "Codebase Analysis Failed - Build ${BUILD_NUMBER}",
                body: """
                Codebase analysis failed.
                
                Build: ${BUILD_NUMBER}
                Branch: ${BRANCH_NAME}
                Logs: ${BUILD_URL}console
                
                Please check the build logs for details.
                """,
                to: "${env.CHANGE_AUTHOR_EMAIL ?: 'dev-team@company.com'}"
            )
        }
    }
}
EOF
```

### GitHub Actions Workflow

Create `.github/workflows/codebase-analysis.yml`:

```yaml
cat > .github/workflows/codebase-analysis.yml << 'EOF'
name: Codebase Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run weekly analysis on Sundays at 2 AM UTC
    - cron: '0 2 * * 0'

env:
  AWS_REGION: us-west-2
  ANALYSIS_OUTPUT_DIR: analysis_results

jobs:
  build-analyzer:
    runs-on: ubuntu-latest
    if: |
      contains(github.event.head_commit.modified, 'Containerfile') ||
      contains(github.event.head_commit.modified, 'requirements.txt') ||
      contains(github.event.head_commit.modified, 'codebase_agent.py') ||
      github.event_name == 'schedule'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Podman
      run: |
        sudo apt-get update
        sudo apt-get -y install podman
    
    - name: Build Analysis Container
      run: |
        podman build -t codebase-analysis-agent:${{ github.sha }} .
        podman save codebase-analysis-agent:${{ github.sha }} | gzip > analysis-agent.tar.gz
    
    - name: Upload Container Artifact
      uses: actions/upload-artifact@v4
      with:
        name: analysis-container
        path: analysis-agent.tar.gz
        retention-days: 1

  analyze-codebase:
    runs-on: ubuntu-latest
    needs: build-analyzer
    if: always() && (needs.build-analyzer.result == 'success' || needs.build-analyzer.result == 'skipped')
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for better analysis
    
    - name: Install Podman
      run: |
        sudo apt-get update
        sudo apt-get -y install podman
    
    - name: Download Container Artifact
      if: needs.build-analyzer.result == 'success'
      uses: actions/download-artifact@v4
      with:
        name: analysis-container
    
    - name: Load Container Image
      if: needs.build-analyzer.result == 'success'
      run: podman load < analysis-agent.tar.gz
    
    - name: Use Pre-built Image
      if: needs.build-analyzer.result == 'skipped'
      run: |
        # For scheduled runs or when container didn't change
        podman build -t codebase-analysis-agent:${{ github.sha }} .
    
    - name: Run Codebase Analysis
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        mkdir -p ${{ env.ANALYSIS_OUTPUT_DIR }}
        
        podman run --rm \
          -v ${{ github.workspace }}:/workspace:ro \
          -v ${{ github.workspace }}/${{ env.ANALYSIS_OUTPUT_DIR }}:/app/analysis_output \
          -e AWS_ACCESS_KEY_ID="${{ secrets.AWS_ACCESS_KEY_ID }}" \
          -e AWS_SECRET_ACCESS_KEY="${{ secrets.AWS_SECRET_ACCESS_KEY }}" \
          -e AWS_DEFAULT_REGION=${{ env.AWS_REGION }} \
          codebase-analysis-agent:${{ github.sha }} \
          python analysis_scripts/batch_analysis.py \
          --repo-path /workspace \
          --output-dir /app/analysis_output
    
    - name: Upload Analysis Results
      uses: actions/upload-artifact@v4
      with:
        name: analysis-results-${{ github.sha }}
        path: ${{ env.ANALYSIS_OUTPUT_DIR }}/
        retention-days: 30
    
    - name: Comment PR with Analysis Summary
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const fs = require('fs');
          const path = '${{ env.ANALYSIS_OUTPUT_DIR }}/analysis_summary.json';
          
          if (fs.existsSync(path)) {
            const summary = JSON.parse(fs.readFileSync(path, 'utf8'));
            const comment = `
          ## 🔍 Codebase Analysis Results
          
          **Analysis Summary:**
          - ✅ Successful tasks: ${summary.analysis_metadata.successful_tasks}
          - ❌ Failed tasks: ${summary.analysis_metadata.failed_tasks}
          - ⏱️ Duration: ${Math.round(summary.analysis_metadata.duration_seconds)}s
          
          **Key Findings:**
          - Business logic entities analyzed
          - Migration documentation generated
          - Risk assessment completed
          
          📋 [View detailed results in artifacts](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})
          `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          }

  security-scan:
    runs-on: ubuntu-latest
    needs: analyze-codebase
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Download Analysis Results
      uses: actions/download-artifact@v4
      with:
        name: analysis-results-${{ github.sha }}
        path: ${{ env.ANALYSIS_OUTPUT_DIR }}
    
    - name: Security Risk Assessment
      run: |
        python3 -c "
        import json
        import sys
        
        # Load business logic analysis
        with open('${{ env.ANALYSIS_OUTPUT_DIR }}/business_logic.json', 'r') as f:
            business_logic = json.load(f)
        
        # Check for high-risk patterns
        high_risk_count = 0
        if isinstance(business_logic, list):
            high_risk_count = len([entity for entity in business_logic 
                                 if isinstance(entity, dict) and entity.get('risk_level') == 'high'])
        
        print(f'High-risk components detected: {high_risk_count}')
        
        if high_risk_count > 5:
            print('⚠️ Too many high-risk components detected!')
            sys.exit(1)
        "

  deploy-results:
    runs-on: ubuntu-latest
    needs: analyze-codebase
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - name: Download Analysis Results
      uses: actions/download-artifact@v4
      with:
        name: analysis-results-${{ github.sha }}
        path: ${{ env.ANALYSIS_OUTPUT_DIR }}
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Deploy to S3
      if: env.S3_RESULTS_BUCKET != ''
      env:
        S3_RESULTS_BUCKET: ${{ secrets.S3_RESULTS_BUCKET }}
      run: |
        aws s3 sync ${{ env.ANALYSIS_OUTPUT_DIR }}/ \
          s3://${S3_RESULTS_BUCKET}/analysis-results/${{ github.sha }}/ \
          --delete
        echo "Results deployed to s3://${S3_RESULTS_BUCKET}/analysis-results/${{ github.sha }}/"
EOF
```

### Incremental Analysis Script

Create the incremental analysis for change detection:

```bash
cat > analysis_scripts/incremental_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Incremental analysis script for CI/CD
Analyzes only changed files and their dependencies
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Set
from codebase_agent import create_codebase_agent, CodebaseAnalyzer

def get_affected_files(changed_files: List[str], repo_path: str) -> Set[str]:
    """Get all files potentially affected by changes"""
    
    affected = set(changed_files)
    repo_path_obj = Path(repo_path)
    
    for changed_file in changed_files:
        file_path = repo_path_obj / changed_file
        
        if not file_path.exists():
            continue
            
        # If it's a Struts config file, include all Action classes
        if 'struts-config.xml' in changed_file:
            java_files = list(repo_path_obj.rglob("*Action.java"))
            affected.update(str(f.relative_to(repo_path_obj)) for f in java_files)
        
        # If it's a Java file, include related files
        elif changed_file.endswith('.java'):
            file_name = Path(changed_file).stem
            
            # Include related test files
            test_files = list(repo_path_obj.rglob(f"*{file_name}*Test.java"))
            affected.update(str(f.relative_to(repo_path_obj)) for f in test_files)
            
            # Include related form classes if it's an Action
            if 'Action' in file_name:
                form_name = file_name.replace('Action', 'Form')
                form_files = list(repo_path_obj.rglob(f"*{form_name}*.java"))
                affected.update(str(f.relative_to(repo_path_obj)) for f in form_files)
    
    return affected

def analyze_changes(repo_path: str, changed_files: List[str], output_dir: str):
    """Run focused analysis on changed files"""
    
    print(f"🔍 Analyzing {len(changed_files)} changed files")
    
    # Get all affected files
    affected_files = get_affected_files(changed_files, repo_path)
    print(f"📁 Total affected files: {len(affected_files)}")
    
    # Create agent
    agent = create_codebase_agent(repo_path)
    
    # Focus analysis on changed components
    if any('.java' in f for f in changed_files):
        print("☕ Java changes detected - analyzing business logic impact")
        
        java_files = [f for f in affected_files if f.endswith('.java')]
        java_list = ', '.join(java_files[:10])  # Limit for prompt length
        if len(java_files) > 10:
            java_list += f" and {len(java_files) - 10} more files"
        
        query = f"""
        Analyze the business logic impact of changes to these Java files: {java_list}
        
        Focus on:
        1. What business logic has been modified?
        2. Are there any new business rules or validation changes?
        3. What are the risk implications of these changes?
        4. Which other components might be affected by these changes?
        5. What testing should be prioritized for these changes?
        
        Provide a concise impact assessment for the development team.
        """
        
        response = agent.run(query)
        
        # Save incremental analysis
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "incremental_analysis.md", 'w') as f:
            f.write(f"# Incremental Analysis - Changes Impact\n\n")
            f.write(f"**Changed Files:** {len(changed_files)}\n")
            f.write(f"**Affected Files:** {len(affected_files)}\n\n")
            f.write(f"## Business Logic Impact Analysis\n\n")
            f.write(response)
        
        print("✅ Incremental analysis completed")
    
    else:
        print("📝 No Java changes detected - skipping business logic analysis")
        
        # Just create a simple change summary
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "change_summary.json", 'w') as f:
            json.dump({
                "changed_files": changed_files,
                "analysis_needed": False,
                "reason": "No Java files modified"
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Incremental codebase analysis")
    parser.add_argument("--repo-path", required=True, help="Repository path")
    parser.add_argument("--changed-files", required=True, help="File containing list of changed files")
    parser.add_argument("--output-dir", default="./analysis_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Read changed files
    changed_files = []
    if os.path.exists(args.changed_files):
        with open(args.changed_files, 'r') as f:
            changed_files = [line.strip() for line in f if line.strip()]
    
    if not changed_files:
        print("No changed files detected - skipping analysis")
        sys.exit(0)
    
    # Run incremental analysis
    try:
        analyze_changes(args.repo_path, changed_files, args.output_dir)
        print("✅ Incremental analysis completed successfully")
    except Exception as e:
        print(f"❌ Incremental analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x analysis_scripts/incremental_analysis.py
```

### Report Generation Script

```bash
cat > analysis_scripts/generate_reports.py << 'EOF'
#!/usr/bin/env python3
"""
Generate HTML reports from analysis results
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def generate_html_report(analysis_dir: str, output_dir: str):
    """Generate comprehensive HTML report"""
    
    analysis_path = Path(analysis_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load analysis results
    results = {}
    for json_file in analysis_path.glob("*.json"):
        with open(json_file, 'r') as f:
            try:
                results[json_file.stem] = json.load(f)
            except json.JSONDecodeError:
                continue
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .risk-high {{ color: #d32f2f; }}
            .risk-medium {{ color: #f57c00; }}
            .risk-low {{ color: #388e3c; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Codebase Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Summary</h2>
    """
    
    # Add summary from analysis_summary.json
    if 'analysis_summary' in results:
        summary = results['analysis_summary']
        html_content += f"""
            <ul>
                <li><strong>Total Tasks:</strong> {summary.get('analysis_metadata', {}).get('total_tasks', 'N/A')}</li>
                <li><strong>Successful:</strong> {summary.get('analysis_metadata', {}).get('successful_tasks', 'N/A')}</li>
                <li><strong>Duration:</strong> {summary.get('analysis_metadata', {}).get('duration_seconds', 'N/A')}s</li>
            </ul>
        """
    
    # Add business logic section
    if 'business_logic' in results:
        business_logic = results['business_logic']
        if isinstance(business_logic, list):
            html_content += f"""
        </div>
        
        <div class="section">
            <h2>🏗️ Business Logic Components ({len(business_logic)})</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Risk Level</th>
                    <th>Complexity</th>
                    <th>File Path</th>
                </tr>
            """
            
            for entity in business_logic[:50]:  # Limit to first 50
                if isinstance(entity, dict):
                    risk_class = f"risk-{entity.get('risk_level', 'unknown')}"
                    html_content += f"""
                <tr>
                    <td>{entity.get('name', 'Unknown')}</td>
                    <td>{entity.get('type', 'Unknown')}</td>
                    <td class="{risk_class}">{entity.get('risk_level', 'Unknown').title()}</td>
                    <td>{entity.get('complexity_score', 'N/A')}</td>
                    <td>{entity.get('file_path', 'Unknown')}</td>
                </tr>
                    """
            
            html_content += "</table>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_path / "analysis_report.html", 'w') as f:
        f.write(html_content)
    
    print(f"📋 HTML report generated: {output_path / 'analysis_report.html'}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports from analysis")
    parser.add_argument("--input-dir", required=True, help="Directory containing analysis results")
    parser.add_argument("--output-dir", help="Output directory for reports")
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = str(Path(args.input_dir) / "reports")
    
    generate_html_report(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
EOF

chmod +x analysis_scripts/generate_reports.py
```

---

## 📚 Additional Resources

- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **Strands Agents GitHub**: https://github.com/strands-agents
- **Java AST Documentation**: https://github.com/c2nes/javalang
- **Struts Documentation**: https://struts.apache.org/

## 💬 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Test with the simple example codebase first
4. Review AWS Bedrock model access status

Ready to modernize your legacy codebase! 🚀