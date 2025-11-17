"""
LangGraph Statistical Validation Test Agent

This agent performs statistical data analysis, validates results,
and generates comprehensive reports with full reasoning transparency.
"""

import os
from typing import TypedDict, Annotated, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import subprocess
import tempfile
import yaml
from pathlib import Path


# Define the state for our statistical analysis agent
class StatisticalAnalysisState(TypedDict):
    # Input
    task: str
    data_file_path: str  # First file (for backward compatibility)
    data_file_paths: list  # All files (for multi-file support)
    output_dir: str  # Directory to save plots and output files

    # Analysis planning
    analysis_objective: str  # Short objective (2-3 sentences)
    analysis_plan: str  # Full analysis plan with methodology, approach, assumptions
    data_summary: str

    # Code generation & execution
    code: str
    execution_result: str
    execution_error: str

    # Report sections
    analysis_details: str
    analysis_conclusions: str

    # Reasoning/thought tracking
    reasoning_log: list

    # Control flow
    iteration: int
    max_iterations: int
    should_continue: bool
    is_valid: bool


# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# Helper function to log agent thoughts
def add_thought(state: StatisticalAnalysisState, node_name: str, thought: str) -> list:
    """Add a reasoning entry to the log."""
    reasoning_entry = {
        "node": node_name,
        "iteration": state.get("iteration", 0),
        "thought": thought,
        "timestamp": datetime.now().isoformat()
    }
    return state.get("reasoning_log", []) + [reasoning_entry]


def clean_code(code: str) -> str:
    """Remove markdown code fences and explanatory text from generated code."""
    code = code.strip()

    # Remove starting fence
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```py"):
        code = code[5:]
    elif code.startswith("```"):
        code = code[3:]

    # Remove ending fence
    if code.endswith("```"):
        code = code[:-3]

    code = code.strip()

    # Remove any remaining ``` lines (sometimes LLM adds them in the middle)
    lines = code.split('\n')
    cleaned_lines = []
    in_code = True

    for line in lines:
        stripped = line.strip()

        # Skip lines that are just markdown fences
        if stripped in ['```', '```python', '```py']:
            continue

        # Skip explanatory text that's clearly not code
        # LLM sometimes adds sentences like "This code will..." at the end
        if stripped and not any([
            stripped.startswith('#'),  # Comment
            stripped.startswith('import '),
            stripped.startswith('from '),
            '=' in line,  # Assignment
            stripped.startswith('def '),
            stripped.startswith('class '),
            stripped.startswith('if '),
            stripped.startswith('for '),
            stripped.startswith('while '),
            stripped.startswith('with '),
            stripped.startswith('try:'),
            stripped.startswith('except'),
            stripped.startswith('finally:'),
            stripped.startswith('return '),
            stripped.startswith('print('),
            stripped.startswith('plt.'),
            stripped.startswith('pd.'),
            stripped.startswith('np.'),
            line.strip().endswith(':'),  # Control structure
            line.strip().endswith(')'),  # Function call
            line.strip().endswith(']'),  # Array/list
            line.strip().endswith('}'),  # Dict
            line == '',  # Empty line
            line[0].isspace() and len(line) > 1  # Indented line (likely in a block)
        ]):
            # This looks like explanatory text, skip it
            continue

        cleaned_lines.append(line)

    code = '\n'.join(cleaned_lines)

    return code.strip()


def understand_data(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """First node: Examine the data file(s) and generate a summary."""

    data_file_paths = state.get("data_file_paths", [state["data_file_path"]])

    # Create file list description
    if len(data_file_paths) == 1:
        files_desc = f"this data file: {data_file_paths[0]}"
    else:
        files_desc = f"these {len(data_file_paths)} data files:\n" + "\n".join(f"  - {fp}" for fp in data_file_paths)

    system_prompt = """You are a data analyst. Examine the provided data file(s) and generate a comprehensive
    data summary. Your summary should include:
    - Data shape (rows, columns) for each file
    - Column names and data types
    - Missing values
    - Basic descriptive statistics
    - Any data quality issues or notable patterns
    - If multiple files: relationships between files (common columns, potential joins)

    Provide a clear, concise summary that will help plan the statistical analysis."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Please examine {files_desc}

Generate a Python code snippet that loads the data and prints a comprehensive summary.
If multiple files are provided, examine each one and note any relationships between them.
Only output the Python code without explanations or markdown formatting.""")
    ]

    response = llm.invoke(messages)
    code = clean_code(response.content)

    # Execute the code to get data summary
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        os.unlink(temp_file)
        data_summary = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        data_summary = f"Error examining data: {str(e)}"

    # Log the reasoning
    file_count = len(data_file_paths)
    files_msg = f"{file_count} file(s): {', '.join(Path(fp).name for fp in data_file_paths)}"

    thought = f"""Examined data {files_msg}

Data Summary Generated:
{data_summary[:500]}...

This summary will guide our analysis approach."""

    reasoning_log = add_thought(state, "understand_data", thought)

    return {
        **state,
        "data_summary": data_summary,
        "reasoning_log": reasoning_log
    }


def plan_analysis(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Second node: Determine the analysis objective and approach."""

    task = state["task"]
    data_summary = state["data_summary"]

    system_prompt = """You are a statistical analyst. Based on the user's task and the data summary,
    determine:
    1. The clear analysis objective (what question are we answering?)
    2. The appropriate statistical methods/tests to use
    3. Any assumptions that need to be checked
    4. Expected outputs (tables, plots, test results)

    Be specific about the statistical approach and explain your reasoning."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Task: {task}

Data Summary:
{data_summary}

Please provide:
1. A clear analysis objective statement
2. The recommended statistical approach with rationale""")
    ]

    response = llm.invoke(messages)
    analysis_plan = response.content.strip()

    # Extract objective (first paragraph or until first section header)
    objective_lines = []
    for line in analysis_plan.split('\n'):
        line_lower = line.lower().strip()
        # Stop at section headers like "### 2." or "Recommended Statistical Approach"
        if line_lower.startswith('###') or line_lower.startswith('## 2') or 'approach' in line_lower or 'method' in line_lower:
            break
        if line.strip():
            objective_lines.append(line)

    analysis_objective = '\n'.join(objective_lines).strip()

    # If objective is empty or too long, just use first few lines
    if not analysis_objective or len(analysis_objective) > 500:
        objective_lines = []
        for line in analysis_plan.split('\n'):
            if line.strip():
                objective_lines.append(line)
                if len(objective_lines) >= 3:
                    break
        analysis_objective = '\n'.join(objective_lines)

    # Check data compatibility with analysis requirements
    compatibility_prompt = """Review the task and available data to check compatibility.

Can the available data support this analysis? Verify:
1. Does the data contain the required columns/variables for the planned methods?
2. Is there sufficient sample size for the statistical tests?
3. Are there any fundamental mismatches between task requirements and available data?

Respond with ONLY:
- "COMPATIBLE: [brief reason]" if the analysis is feasible with available data
- "INCOMPATIBLE: [explain what specific data/columns are missing]" if data is insufficient"""

    compatibility_messages = [
        SystemMessage(content=compatibility_prompt),
        HumanMessage(content=f"""Task: {task}

Available Data Summary:
{data_summary}

Planned Analysis:
{analysis_plan}

Assess data compatibility.""")
    ]

    compatibility_response = llm.invoke(compatibility_messages)
    compatibility_assessment = compatibility_response.content.strip()

    # If data is incompatible, prepend warning to objective
    # Check for INCOMPATIBLE first (to avoid substring match with COMPATIBLE)
    data_compatible = not compatibility_assessment.upper().startswith("INCOMPATIBLE")
    if not data_compatible:
        analysis_objective = f"⚠️ DATA COMPATIBILITY ISSUE:\n{compatibility_assessment}\n\n{analysis_objective}"

    # Log the reasoning
    thought = f"""Task Interpretation: {task}

Full Analysis Plan Generated:
{analysis_plan}

Data Compatibility Check:
{compatibility_assessment}
Status: {"✓ Compatible" if data_compatible else "✗ Incompatible - analysis may fail"}

Extracted Objective (2-3 sentences):
{analysis_objective}

This complete plan guides what statistical methods we'll implement."""

    reasoning_log = add_thought(state, "plan_analysis", thought)

    return {
        **state,
        "analysis_objective": analysis_objective,
        "analysis_plan": analysis_plan,  # Store the FULL plan
        "reasoning_log": reasoning_log
    }


def write_analysis_code(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Third node: Generate Python code to perform the statistical analysis."""

    iteration = state.get("iteration", 0)
    task = state["task"]
    data_file_paths = state.get("data_file_paths", [state["data_file_path"]])
    output_dir = state.get("output_dir", ".")
    data_summary = state["data_summary"]
    analysis_objective = state["analysis_objective"]

    # Extract just the filenames since code runs with cwd=output_dir
    from pathlib import Path
    data_filenames = [Path(fp).name for fp in data_file_paths]

    # For single file, use simpler description
    if len(data_filenames) == 1:
        files_instruction = f"Data File: {data_filenames[0]}\n(This file is in the current working directory, so just use this filename directly)"
    else:
        files_instruction = f"Data Files: {', '.join(data_filenames)}\n(These files are in the current working directory, so just use these filenames directly)"

    if iteration == 0:
        # First iteration - write initial code
        system_prompt = """You are an expert statistical programmer. Write Python code to perform statistical analysis.

Requirements:
- Include ALL necessary imports at the top (os, pandas, numpy, matplotlib, scipy, etc.)
- Load the data from the specified file path (file is in current working directory)
- Perform appropriate statistical tests and validation
- Check statistical assumptions (normality, homogeneity, independence, etc.)
- Generate informative visualizations and SAVE ALL PLOTS to current directory
- **CRITICAL**: Save plots with descriptive filenames (e.g., plt.savefig('distribution_plot.png'))
- DO NOT use plt.show() - only plt.savefig()
- Print structured results that include:
  * Test statistics and p-values
  * Confidence intervals where appropriate
  * Effect sizes
  * Assumption check results
- Handle missing data appropriately
- Use libraries like pandas, numpy, scipy, statsmodels, matplotlib, seaborn

**CRITICAL FILE HANDLING:**
1. ALWAYS sanitize column names before using in filenames:
   - Replace special characters (/, \\, :, *, ?, ", <, >, |, spaces) with underscores
   - Example: column "Close/Last" must become "Close_Last" in filenames
   - Use: safe_name = col.replace('/', '_').replace('\\', '_').replace(' ', '_')
2. Check if output_dir variable exists in scope, if yes use f"{output_dir}/filename.png"
3. Never use column names directly in file paths without sanitizing

Only output the Python code without explanations or markdown formatting.
The code should be production-ready and include error handling."""

        # Get the full analysis plan
        analysis_plan = state.get("analysis_plan", analysis_objective)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Task: {task}

{files_instruction}

Data Summary:
{data_summary}

Full Analysis Plan:
{analysis_plan}

Write complete Python code to perform this analysis following the plan above.
IMPORTANT: Save all plots to the current directory with descriptive names like 'var_distribution.png', 'rolling_var.png', etc.""")
        ]
    else:
        # Subsequent iterations - improve based on validation feedback
        system_prompt = """You are an expert statistical programmer. Based on the validation feedback,
improve the analysis code. Fix any errors, address failed assumptions, or enhance the analysis.

**CRITICAL FILE HANDLING:**
1. Include ALL necessary imports (os, pandas, numpy, matplotlib, scipy, etc.)
2. ALWAYS sanitize column names before using in filenames:
   - Replace special characters (/, \\, :, *, ?, ", <, >, |, spaces) with underscores
   - Example: "Close/Last" → "Close_Last"
   - Use: safe_name = col.replace('/', '_').replace('\\', '_').replace(' ', '_')
3. Never use column names directly in file paths without sanitizing

Only output the Python code without explanations or markdown formatting."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Task: {task}

Previous Code:
{state['code']}

Execution Result:
{state['execution_result']}

Execution Error (if any):
{state['execution_error']}

Validation Feedback:
{state.get('analysis_details', 'No specific feedback yet')}

Please write improved code based on this feedback.""")
        ]

    response = llm.invoke(messages)
    code = clean_code(response.content)

    # Log the reasoning
    if iteration == 0:
        files_msg = f"{len(data_filenames)} file(s): {', '.join(data_filenames)}"
        thought = f"""Generated initial analysis code.

Approach:
- Loading data from {files_msg}
- Implementing statistical tests based on the analysis plan
- Including assumption checks and validation
- Creating visualizations for results

Code length: {len(code)} characters
Libraries used: pandas, numpy, scipy, matplotlib (inferred from task)"""
    else:
        thought = f"""Iteration {iteration}: Revised analysis code.

Improvements made:
- Addressed execution errors or validation feedback
- Enhanced statistical rigor
- Improved error handling

Code length: {len(code)} characters"""

    reasoning_log = add_thought(state, "write_analysis_code", thought)

    return {
        **state,
        "code": code,
        "iteration": iteration + 1,
        "reasoning_log": reasoning_log
    }


def execute_code(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Fourth node: Execute the generated analysis code and capture the results."""

    code = state["code"]
    output_dir = state.get("output_dir", ".")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Execute the code with the output directory as the working directory
        # This ensures relative paths work and plots are saved in the right place
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=60,  # Increased timeout for statistical computations
            cwd=output_dir  # Set working directory to output directory
        )

        # Clean up temp file
        os.unlink(temp_file)

        execution_result = result.stdout
        execution_error = result.stderr if result.returncode != 0 else ""

        # Log the reasoning
        if execution_error:
            thought = f"""Code execution failed.

Error:
{execution_error[:300]}...

Will need to revise the code to fix these issues."""
        else:
            thought = f"""Code executed successfully.

Output preview:
{execution_result[:500]}...

Analysis produced results. Now validating statistical assumptions and completeness."""

        reasoning_log = add_thought(state, "execute_code", thought)

        return {
            **state,
            "execution_result": execution_result,
            "execution_error": execution_error,
            "reasoning_log": reasoning_log
        }

    except subprocess.TimeoutExpired:
        thought = "Code execution timed out after 60 seconds. Analysis may be too computationally intensive or contains infinite loops."
        reasoning_log = add_thought(state, "execute_code", thought)
        return {
            **state,
            "execution_result": "",
            "execution_error": "Code execution timed out after 60 seconds",
            "reasoning_log": reasoning_log
        }
    except Exception as e:
        thought = f"Unexpected error during execution: {str(e)}"
        reasoning_log = add_thought(state, "execute_code", thought)
        return {
            **state,
            "execution_result": "",
            "execution_error": f"Error during execution: {str(e)}",
            "reasoning_log": reasoning_log
        }


def validate_results(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Fifth node: Validate the analysis results and determine if improvements are needed."""

    system_prompt = """You are a statistical validator. Review the analysis code and results to determine:

1. EXECUTION STATUS:
   - Did the code execute successfully?
   - Are there any errors that need fixing?

2. STATISTICAL VALIDITY:
   - Were appropriate statistical tests used?
   - Were assumptions checked (normality, independence, homogeneity, etc.)?
   - Are the assumptions met? If not, were appropriate alternatives used?
   - Are effect sizes and confidence intervals reported?

3. COMPLETENESS:
   - Does the analysis fully address the task?
   - Are results interpretable and clearly presented?
   - Are visualizations appropriate?

Respond with ONLY:
- "Analysis is valid and complete." if everything is correct
- OR specific improvements needed (be brief and actionable)

Do NOT generate report content - only validation feedback."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Task: {state['task']}

Code:
{state['code']}

Execution Result:
{state['execution_result']}

Execution Error (if any):
{state['execution_error']}

Please provide your validation assessment.""")
    ]

    response = llm.invoke(messages)
    validation = response.content.strip()

    # Determine if analysis is valid and complete
    is_valid = (
        "valid and complete" in validation.lower() or
        "no improvements needed" in validation.lower()
    ) and state["execution_error"] == ""

    # Determine if we should continue iterating
    should_continue = (
        state["iteration"] < state["max_iterations"] and
        not is_valid
    )

    # If there's an error, we should try to fix it
    if state["execution_error"] and state["iteration"] < state["max_iterations"]:
        should_continue = True

    # Log the reasoning
    thought = f"""Validation Assessment:

Execution Status: {"Success" if not state["execution_error"] else "Failed"}
Statistical Validity: {"Valid" if is_valid else "Needs improvement"}
Should Continue: {should_continue}

Validation Details:
{validation[:500]}...

{"Analysis meets all criteria and is ready for reporting." if is_valid else "Further iteration needed to address issues."}"""

    reasoning_log = add_thought(state, "validate_results", thought)

    return {
        **state,
        "is_valid": is_valid,
        "should_continue": should_continue,
        "reasoning_log": reasoning_log
    }


def compile_report_content(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """NEW Sixth node: Compile comprehensive, self-contained report content from analysis results.

    This is separate from validation - it extracts detailed report sections with actual data and results.
    """

    # GUARD: Don't compile report if execution failed or produced no output
    execution_error = state.get('execution_error', '')
    execution_result = state.get('execution_result', '').strip()

    if execution_error or not execution_result or len(execution_result) < 50:
        # Check if there was a data compatibility issue
        analysis_objective = state.get('analysis_objective', '')
        has_compatibility_issue = '⚠️ DATA COMPATIBILITY ISSUE' in analysis_objective

        thought = f"""Cannot compile report - analysis execution failed or produced no meaningful output.

Execution Error: {execution_error if execution_error else 'None'}
Execution Output Length: {len(execution_result)} characters
Data Compatibility Issue Detected: {has_compatibility_issue}

The analysis did not complete successfully. Report compilation skipped."""

        reasoning_log = add_thought(state, "compile_report_content", thought)

        # Preserve compatibility warning if it exists
        if has_compatibility_issue:
            # Extract the compatibility message
            compatibility_msg = analysis_objective.split('\n\n')[0] if '\n\n' in analysis_objective else analysis_objective

            details_msg = f"{compatibility_msg}\n\nThe analysis could not proceed due to insufficient or incompatible data. The provided dataset does not contain the required variables or structure needed for the requested analysis."
            conclusions_msg = f"Analysis cannot be completed - data compatibility issue prevents execution of the planned statistical methods."
        else:
            details_msg = "Analysis could not be completed. The code either failed to execute or did not produce valid results. Please check that the data file is accessible and compatible with the requested analysis."
            conclusions_msg = "No conclusions available - analysis execution failed."

        return {
            **state,
            "analysis_details": details_msg,
            "analysis_conclusions": conclusions_msg,
            "is_valid": False,  # Mark as invalid since analysis failed
            "reasoning_log": reasoning_log
        }

    system_prompt = """You are a report compiler. Create a comprehensive, self-contained analysis report.

Extract and format EXACTLY THREE sections based on the ACTUAL RESULTS. Keep sections STRICTLY SEPARATE:

=== SECTION 1: OBJECTIVE ===
(Short and Clear - 2-3 sentences ONLY):
- Clearly describe what the analysis aims to achieve
- What specific question/hypothesis is being tested
DO NOT include methods, results, or conclusions here!

=== SECTION 2: ANALYSIS DETAILS ===
(Comprehensive methodology and results):
* use Markdown or HTML formatting for clarity
* Description of input data (actual sample size, variables examined)
* Statistical methods and tests performed (name SPECIFIC tests used)
* Actual test results with NUMBERS (test statistics, p-values, confidence intervals)
* Visualizations or .png files created (list actual plot filenames)
* Key numeric results in tables or bullet points
* IMPORTANT: Use asterisk (*) for bullet points, NOT dashes (-)
DO NOT include the objective or final conclusions here!

=== SECTION 3: CONCLUSIONS ===
(Answer the objective - keep brief):
* Direct answer to the research question stated in objective
* Statistical evidence supporting conclusion (cite specific p-values/test results)
* Practical interpretation of results
* Key limitations (1-2 sentences)
* IMPORTANT: Use asterisk (*) for bullet points, NOT dashes (-)
DO NOT repeat methodology details here!

CRITICAL FORMATTING RULES:
- Start each section with its header (=== SECTION N: NAME ===)
- Use ACTUAL values from execution output
- Keep sections cleanly separated - NO mixing of content
- Make the report self-contained"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Task: {state['task']}

Full Analysis Plan (with methodology and approach):
{state.get('analysis_plan', state['analysis_objective'])}

Data Summary:
{state['data_summary'][:500]}...

Execution Results (ACTUAL OUTPUT):
{state['execution_result']}

Code Context (for understanding what was done):
{state['code'][:1000]}...

Based on the ACTUAL RESULTS above and the original analysis plan, compile the three report sections.""")
    ]

    response = llm.invoke(messages)
    report_content = response.content.strip()

    # Parse into sections
    lines = report_content.split('\n')
    objective = []
    details = []
    conclusions = []
    current_section = None

    for line in lines:
        line_lower = line.lower()
        if ('objective' in line_lower or '1.' in line) and ':' in line and not current_section:
            current_section = 'objective'
            continue
        elif ('analysis details' in line_lower or 'methodology' in line_lower or '2.' in line) and ':' in line:
            current_section = 'details'
            continue
        elif ('conclusion' in line_lower or '3.' in line) and ':' in line:
            current_section = 'conclusions'
            continue

        if current_section == 'objective':
            objective.append(line)
        elif current_section == 'details':
            details.append(line)
        elif current_section == 'conclusions':
            conclusions.append(line)

    objective_str = '\n'.join(objective).strip()
    details_str = '\n'.join(details).strip()
    conclusions_str = '\n'.join(conclusions).strip()

    # Fallback: if parsing failed, try simpler approach
    if not details_str or not conclusions_str:
        parts = report_content.split('\n\n')
        if len(parts) >= 3:
            objective_str = parts[0] if not objective_str else objective_str
            details_str = parts[1] if not details_str else details_str
            conclusions_str = '\n\n'.join(parts[2:]) if not conclusions_str else conclusions_str

    # Log reasoning
    thought = f"""Compiled comprehensive report content from analysis results.

Report Sections Generated:
- Objective: {len(objective_str)} characters
- Analysis Details: {len(details_str)} characters
- Conclusions: {len(conclusions_str)} characters

Report includes actual test results, p-values, and references to generated plots.
The report is self-contained and can be understood independently."""

    reasoning_log = add_thought(state, "compile_report_content", thought)

    return {
        **state,
        "analysis_objective": objective_str if objective_str else state["analysis_objective"],
        "analysis_details": details_str if details_str else "Analysis details not extracted",
        "analysis_conclusions": conclusions_str if conclusions_str else "Conclusions not extracted",
        "reasoning_log": reasoning_log
    }


def generate_report(state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Seventh node: Generate the final report and reasoning files."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reasoning_log = state.get("reasoning_log", [])

    # ========================================================================
    # 1. Load template and fill with analysis data
    # ========================================================================

    # Find template file - look in current directory and parent directory
    template_path = None
    current_dir = Path.cwd()

    # Try current directory first
    if (current_dir / "analysis_template.yml").exists():
        template_path = current_dir / "analysis_template.yml"
    # Try parent directory
    elif (current_dir.parent / "analysis_template.yml").exists():
        template_path = current_dir.parent / "analysis_template.yml"
    # Try the script's directory
    elif (Path(__file__).parent / "analysis_template.yml").exists():
        template_path = Path(__file__).parent / "analysis_template.yml"

    if template_path and template_path.exists():
        # Load template as YAML
        with open(template_path, 'r') as f:
            report_data = yaml.safe_load(f)

        # Generate title from task
        task = state.get('task', 'Statistical Analysis')

        # If this is a refinement, extract the original task for a cleaner title
        if 'ORIGINAL TASK:' in task:
            # Extract text between "ORIGINAL TASK:" and the next section
            task_for_title = task.split('ORIGINAL TASK:')[1].split('\n\n')[0].strip()
        else:
            task_for_title = task

        title = task_for_title.split('.')[0][:60]
        if len(task_for_title.split('.')[0]) > 60:
            title += "..."

        # Create label from title
        label = "analysis:" + title.lower().replace(' ', '_').replace(',', '').replace(':', '').replace('(', '').replace(')', '')[:40]

        # Fill in the template with actual values
        report_data['title'] = title
        report_data['label'] = label
        report_data['objective'] = state.get("analysis_objective", "Not specified")
        report_data['details'] = state.get("analysis_details", "No details available")
        report_data['conclusion'] = state.get("analysis_conclusions", "No conclusions available")

        # Configure YAML to use block style for multiline strings (more readable)
        # Use a custom Dumper class to avoid global representer issues
        class BlockDumper(yaml.SafeDumper):
            pass

        def str_representer(dumper, data):
            if '\n' in data:
                # Use literal block style (|) for multiline strings
                # PyYAML will automatically add - or + based on trailing whitespace
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        BlockDumper.add_representer(str, str_representer)

        # Save as YAML with proper formatting
        report_content = yaml.dump(report_data, Dumper=BlockDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)
    else:
        # Fallback: build YAML manually if template not found
        print("Warning: analysis_template.yml not found, using fallback format")

        task = state.get('task', 'Statistical Analysis')
        title = task.split('.')[0][:60]
        if len(task.split('.')[0]) > 60:
            title += "..."

        label = "label:" + title.lower().replace(' ', '_').replace(',', '').replace(':', '')[:40]

        report_data = {
            'analysis_title': title,
            'analysis_label': label,
            'analysis_objective': state.get("analysis_objective", "Not specified"),
            'analysis_details': state.get("analysis_details", "No details available"),
            'analysis_conclusions': state.get("analysis_conclusions", "No conclusions available")
        }

        # Configure YAML to use block style for multiline strings (more readable)
        # Use a custom Dumper class to avoid global representer issues
        class BlockDumper(yaml.SafeDumper):
            pass

        def str_representer(dumper, data):
            if '\n' in data:
                # Use literal block style (|) for multiline strings
                # PyYAML will automatically add - or + based on trailing whitespace
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        BlockDumper.add_representer(str, str_representer)

        report_content = yaml.dump(report_data, Dumper=BlockDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ========================================================================
    # 2. Build the AGENT REASONING file (separate)
    # ========================================================================
    reasoning_lines = []
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("AGENT REASONING PROCESS")
    reasoning_lines.append(f"Generated: {timestamp}")
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("")
    reasoning_lines.append("This file contains the complete thought process of the statistical")
    reasoning_lines.append("analysis agent as it worked through the analysis task.")
    reasoning_lines.append("")
    reasoning_lines.append("=" * 80)
    reasoning_lines.append("")

    # Group by iteration
    iterations = {}
    for entry in reasoning_log:
        iter_num = entry.get("iteration", 0)
        if iter_num not in iterations:
            iterations[iter_num] = []
        iterations[iter_num].append(entry)

    for iter_num in sorted(iterations.keys()):
        if iter_num == 0:
            reasoning_lines.append("INITIAL ANALYSIS (Iteration 0):")
        else:
            reasoning_lines.append(f"\nITERATION {iter_num}:")
        reasoning_lines.append("-" * 80)

        for entry in iterations[iter_num]:
            node_name = entry.get("node", "unknown")
            thought = entry.get("thought", "")
            timestamp_entry = entry.get("timestamp", "")

            reasoning_lines.append(f"\n[{node_name}] - {timestamp_entry}")
            reasoning_lines.append("")
            # Indent the thought content
            for line in thought.split('\n'):
                reasoning_lines.append(f"  {line}")
            reasoning_lines.append("")

    reasoning_lines.append("=" * 80)
    reasoning_content = '\n'.join(reasoning_lines)

    # ========================================================================
    # 3. Save all three files to output_dir
    # ========================================================================

    output_dir = state.get("output_dir", ".")
    output_path = Path(output_dir)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the analysis report as YAML
    report_filename = "analysis_report.yml"
    report_path = output_path / report_filename
    try:
        with open(report_path, 'w') as f:
            f.write(report_content)
        report_saved = True
    except Exception as e:
        print(f"Warning: Could not save report to file: {e}")
        report_saved = False

    # Save the reasoning log
    reasoning_filename = "agent_reasoning.txt"
    reasoning_path = output_path / reasoning_filename
    try:
        with open(reasoning_path, 'w') as f:
            f.write(reasoning_content)
        reasoning_saved = True
    except Exception as e:
        print(f"Warning: Could not save reasoning to file: {e}")
        reasoning_saved = False

    # Save the code
    code_filename = "analysis_code.py"
    code_path = output_path / code_filename
    try:
        with open(code_path, 'w') as f:
            f.write(state.get("code", ""))
        code_saved = True
    except Exception as e:
        print(f"Warning: Could not save code to file: {e}")
        code_saved = False

    # Log the final reasoning
    thought = f"""Generated final outputs and saved to files.

Files created:
1. {report_filename} - Analysis report with 3 sections (objective, details, conclusions)
   Status: {'Success' if report_saved else 'Failed'}

2. {reasoning_filename} - Complete agent reasoning process
   Status: {'Success' if reasoning_saved else 'Failed'}
   Contains: {len(reasoning_log)} thought entries across {len(iterations)} iteration(s)

3. {code_filename} - Python analysis code
   Status: {'Success' if code_saved else 'Failed'}"""

    reasoning_log_updated = add_thought(state, "generate_report", thought)

    return {
        **state,
        "reasoning_log": reasoning_log_updated
    }


def should_continue_decision(state: StatisticalAnalysisState) -> Literal["write_analysis_code", "compile_report_content"]:
    """Determine if we should continue iterating or move to report compilation."""
    if state["should_continue"]:
        return "write_analysis_code"
    return "compile_report_content"


# Build the graph
def create_statistical_agent():
    """Create and compile the LangGraph statistical analysis agent."""

    workflow = StateGraph(StatisticalAnalysisState)

    # Add nodes
    workflow.add_node("understand_data", understand_data)
    workflow.add_node("plan_analysis", plan_analysis)
    workflow.add_node("write_analysis_code", write_analysis_code)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("validate_results", validate_results)
    workflow.add_node("compile_report_content", compile_report_content)  # NEW: compile report before generating files
    workflow.add_node("generate_report", generate_report)

    # Add edges - linear flow from start through validation
    workflow.set_entry_point("understand_data")
    workflow.add_edge("understand_data", "plan_analysis")
    workflow.add_edge("plan_analysis", "write_analysis_code")
    workflow.add_edge("write_analysis_code", "execute_code")
    workflow.add_edge("execute_code", "validate_results")

    # Add conditional edge for iteration or completion
    workflow.add_conditional_edges(
        "validate_results",
        should_continue_decision,
        {
            "write_analysis_code": "write_analysis_code",  # Loop back to retry
            "compile_report_content": "compile_report_content"  # Move to report compilation
        }
    )

    # After compiling report content, generate the report files
    workflow.add_edge("compile_report_content", "generate_report")

    # Report generation is the final step
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def run_statistical_analysis(task: str, data_file_path: str, output_dir: str = ".", max_iterations: int = 3):
    """Run the statistical analysis agent on a given task and data file.

    Args:
        task: The analysis task/question to answer
        data_file_path: Path to the data file to analyze
        output_dir: Directory to save plots and output files (default: current directory)
        max_iterations: Maximum number of code improvement iterations (default: 3)

    Returns:
        Final state dictionary containing all analysis results
    """

    # Create the agent
    agent = create_statistical_agent()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize state
    initial_state = {
        "task": task,
        "data_file_path": data_file_path,  # For backward compatibility
        "data_file_paths": [data_file_path] if isinstance(data_file_path, str) else data_file_path,  # Support both single and multiple
        "output_dir": output_dir,
        "analysis_objective": "",
        "analysis_plan": "",  # Full analysis plan with methodology
        "data_summary": "",
        "code": "",
        "execution_result": "",
        "execution_error": "",
        "analysis_details": "",
        "analysis_conclusions": "",
        "reasoning_log": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "should_continue": True,
        "is_valid": False
    }

    # Run the agent
    print(f"Starting Statistical Analysis Agent")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Data: {data_file_path}")
    print(f"Max iterations: {max_iterations}")
    print("=" * 80)
    print("\nAgent is working...\n")

    final_state = agent.invoke(initial_state)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total iterations: {final_state['iteration']}")
    print(f"Analysis valid: {final_state['is_valid']}")
    print(f"\nOutputs generated:")
    print("  1. analysis_report.txt - Analysis report (3 sections)")
    print("  2. agent_reasoning.txt - Complete agent thought process")
    print("  3. analysis_code.py - Python code used for analysis")

    if final_state['execution_error']:
        print(f"\nWarning: Final execution had errors:")
        print(final_state['execution_error'][:200])

    print("\n" + "=" * 80)

    return final_state


if __name__ == "__main__":
    # Example usage

    # Make sure to set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"

    # Example: Statistical analysis
    print("\n\nExample: Value at Risk (VaR) Analysis\n")

    # Note: You'll need to provide an actual data file path
    # For this example, we'll create a synthetic dataset first
    print("Creating sample data file for demonstration...\n")

    # Create a sample CSV file
    import pandas as pd
    import numpy as np

    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    returns = np.random.normal(0.0005, 0.02, 1000)  # Mean return and volatility
    df = pd.DataFrame({
        'date': dates,
        'returns': returns
    })
    sample_data_path = 'sample_returns.csv'
    df.to_csv(sample_data_path, index=False)
    print(f"Sample data saved to: {sample_data_path}\n")

    # Run the statistical analysis
    run_statistical_analysis(
        task="Calculate the 99% Value at Risk (VaR) using both parametric and historical methods. Compare the results and create visualizations showing the return distribution, VaR threshold, and rolling VaR over time.",
        data_file_path=sample_data_path,
        max_iterations=3
    )

    print("\n\nAnalysis complete! Check the following files:")
    print("  1. analysis_report.txt - Final analysis report (3 sections)")
    print("  2. agent_reasoning.txt - Agent's complete thought process")
    print("  3. analysis_code.py - Python code used")
    print("  4. Any generated plots/figures")