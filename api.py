"""
FastAPI wrapper for the Statistical Validation Test Agent

Features:
- Create new analysis sessions
- Stream reasoning steps in real-time
- Refine analysis with follow-up queries
- Retrieve result files
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import asyncio
import json
from datetime import datetime
from pathlib import Path
import shutil
import os

# Import our statistical agent (v2 with improved reporting)
from somesimpleagent_v2 import create_statistical_agent, StatisticalAnalysisState

# Initialize FastAPI
app = FastAPI(
    title="Statistical Analysis Agent API",
    description="AI-powered statistical analysis with reasoning transparency",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="static")

# Read API base URL from environment variable
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:9002')

# Code approval setting - if True, user must approve code before execution
os.environ['REQUIRE_CODE_APPROVAL'] = 'false'
REQUIRE_CODE_APPROVAL = os.getenv('REQUIRE_CODE_APPROVAL', 'false').lower() == 'true'

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

# Directory for storing session files
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# Session metadata file for persistence
SESSION_METADATA_FILE = SESSIONS_DIR / "sessions_metadata.json"


def save_session_metadata():
    """Save session metadata to disk for persistence."""
    metadata = {}
    for sid, session in sessions.items():
        # Only save metadata, not full state
        metadata[sid] = {
            "session_id": sid,
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "task": session["task"],
            "data_file": session.get("data_file", ""),
            "session_dir": session["session_dir"],
            "refinement_count": session.get("refinement_count", 0),
            "artifact_version": session.get("artifact_version", 0),  # Persist version counter
            "require_code_approval": session.get("require_code_approval", False),  # Persist code approval setting
            "feedback_link": session.get("feedback_link"),  # Persist feedback link
            "gitlab_repo": session.get("gitlab_repo"),  # Persist GitLab repo URL
            "gitlab_branch": session.get("gitlab_branch")  # Persist GitLab branch
        }

    with open(SESSION_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_session_metadata():
    """Load session metadata from disk."""
    if SESSION_METADATA_FILE.exists():
        with open(SESSION_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def generate_feedback_link(session_id: str) -> str:
    """
    Generate a feedback link for a completed analysis session.

    PLACEHOLDER FUNCTION - Customize this to generate your feedback link.

    Examples:
    - Google Forms: return f"https://forms.google.com/...?session={session_id}"
    - Typeform: return f"https://form.typeform.com/...#session={session_id}"
    - Custom survey: return f"https://your-domain.com/feedback?session={session_id}"

    Args:
        session_id: The session ID to include in the feedback link

    Returns:
        The feedback URL as a string
    """
    # CUSTOMIZE THIS URL TO YOUR FEEDBACK FORM
    # Example with session ID embedded:
    return f"https://example.com/feedback?session_id={session_id}"


# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisRequest(BaseModel):
    task: str
    max_iterations: int = 3


class RefineRequest(BaseModel):
    refinement_prompt: str
    max_iterations: int = 2


class GitLabPushRequest(BaseModel):
    gitlab_repo: str
    gitlab_branch: str = "main"


class SessionInfo(BaseModel):
    session_id: str
    status: str
    created_at: str
    task: str
    iteration: int
    is_valid: bool


class ReasoningEvent(BaseModel):
    session_id: str
    node: str
    iteration: int
    thought: str
    timestamp: str


# ============================================================================
# Helper Functions
# ============================================================================

def create_session_directory(session_id: str) -> Path:
    """Create a directory for session files."""
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    return session_dir


def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve a session or raise 404."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


async def stream_reasoning(session_id: str, state: StatisticalAnalysisState):
    """Stream reasoning events as Server-Sent Events."""
    reasoning_log = state.get("reasoning_log", [])

    for entry in reasoning_log:
        event = {
            "session_id": session_id,
            "node": entry.get("node", "unknown"),
            "iteration": entry.get("iteration", 0),
            "thought": entry.get("thought", ""),
            "timestamp": entry.get("timestamp", "")
        }
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.1)  # Small delay for streaming effect


def run_agent_with_streaming(session_id: str, state: StatisticalAnalysisState) -> StatisticalAnalysisState:
    """Run the agent and update session with reasoning events in real-time."""
    from langgraph.pregel import Pregel

    agent = create_statistical_agent()
    session_dir = create_session_directory(session_id)

    # Get version number for this run
    version = sessions[session_id].get("artifact_version", 0) + 1
    sessions[session_id]["artifact_version"] = version
    print(f"  Running analysis version {version} for session {session_id[:8]}...")

    # Stream execution and update session state in real-time
    final_state = None
    last_node_name = None

    for event in agent.stream(state):
        # Check if session has been cancelled
        if sessions[session_id]["status"] == "cancelled":
            print(f"  Session {session_id[:8]} cancelled by user")
            return state  # Return current state without completing

        # Update session state with latest event
        if event:
            # Extract state from event
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    # Update session with latest state
                    sessions[session_id]["state"] = node_state
                    sessions[session_id]["updated_at"] = datetime.now().isoformat()
                    final_state = node_state
                    last_node_name = node_name

                    # If code was just generated and approval is required, wait for approval
                    # Use session-level setting instead of global REQUIRE_CODE_APPROVAL
                    session_requires_approval = sessions[session_id].get("require_code_approval", False)
                    if node_name == "write_analysis_code" and session_requires_approval:
                        print(f"  Code approval required for session {session_id[:8]}")
                        sessions[session_id]["status"] = "awaiting_approval"
                        sessions[session_id]["approval_status"] = "pending"
                        sessions[session_id]["pending_code"] = node_state.get("code", "")
                        sessions[session_id]["updated_at"] = datetime.now().isoformat()
                        save_session_metadata()

                        # Wait for approval (with timeout)
                        max_wait_seconds = 300  # 5 minutes timeout
                        wait_interval = 0.5
                        elapsed = 0

                        while elapsed < max_wait_seconds:
                            import time
                            time.sleep(wait_interval)
                            elapsed += wait_interval

                            approval_status = sessions[session_id].get("approval_status", "pending")

                            if approval_status == "approved":
                                print(f"  Code approved for session {session_id[:8]}")
                                sessions[session_id]["status"] = "processing"
                                break
                            elif approval_status == "rejected" or sessions[session_id]["status"] == "cancelled":
                                print(f"  Code rejected or cancelled for session {session_id[:8]}")
                                return state  # Stop execution

                        if elapsed >= max_wait_seconds:
                            print(f"  Approval timeout for session {session_id[:8]}")
                            sessions[session_id]["status"] = "cancelled"
                            return state

    # Check one more time before finalizing
    if sessions[session_id]["status"] == "cancelled":
        print(f"  Session {session_id[:8]} cancelled by user")
        return state

    if final_state is None:
        final_state = agent.invoke(state)

    # Validate that agent actually ran before marking as completed
    reasoning_log = final_state.get("reasoning_log", [])
    has_code = bool(final_state.get("code", "").strip())
    has_analysis = bool(final_state.get("analysis_details", "").strip())

    # Check if agent actually did work
    if not reasoning_log or len(reasoning_log) == 0:
        # No nodes executed - agent never ran
        print(f"  ERROR: Agent never executed any nodes for session {session_id[:8]}")
        sessions[session_id]["state"] = final_state
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = "Agent workflow did not execute. Check API key and configuration."
        sessions[session_id]["updated_at"] = datetime.now().isoformat()
        save_session_metadata()
        return final_state

    # Update session
    sessions[session_id]["state"] = final_state
    sessions[session_id]["status"] = "completed"
    sessions[session_id]["updated_at"] = datetime.now().isoformat()

    # NOTE: Feedback link is NOT generated here automatically
    # This allows you to generate it asynchronously after completion
    # Use the /analysis/{session_id}/set_feedback_link endpoint to set it when ready
    # Or uncomment the lines below to generate it immediately:
    # feedback_link = generate_feedback_link(session_id)
    # sessions[session_id]["feedback_link"] = feedback_link
    # print(f"  Generated feedback link: {feedback_link}")

    # The agent already saved files to session_dir
    # Create versioned copies for history
    version_suffix = f"_v{version}"

    # Files the agent created
    base_files = [
        ("analysis_report.yml", f"analysis_report{version_suffix}.yml"),
        ("agent_reasoning.txt", f"agent_reasoning{version_suffix}.txt"),
        ("analysis_code.py", f"analysis_code{version_suffix}.py")
    ]

    # Copy to versioned names
    for base_name, versioned_name in base_files:
        base_path = session_dir / base_name
        versioned_path = session_dir / versioned_name
        if base_path.exists():
            shutil.copy2(base_path, versioned_path)

    print(f"  Created versioned copies as v{version} for all artifacts")

    # Save session metadata
    save_session_metadata()

    return final_state




# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web UI with dynamic API_BASE configuration."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_base_url": API_BASE_URL
    })


@app.get("/api/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Statistical Analysis Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analysis/new": "Start a new analysis",
            "POST /analysis/{session_id}/refine": "Refine an existing analysis",
            "GET /analysis/{session_id}/status": "Get analysis status",
            "GET /analysis/{session_id}/stream": "Stream reasoning events (SSE)",
            "GET /analysis/{session_id}/files/{filename}": "Download result files",
            "DELETE /analysis/{session_id}": "Delete a session"
        }
    }


@app.post("/analysis/new")
async def create_analysis(
    files: List[UploadFile] = File(...),
    task: str = Form(...),
    max_iterations: int = Form(3),
    require_code_approval: str = Form("false"),
    gitlab_repo: str = Form(None),
    gitlab_branch: str = Form("main")
):
    """
    Start a new statistical analysis session.

    - Upload data file(s) (CSV, Excel, etc.)
    - Provide an analysis task/question
    - Optionally require code approval before execution
    - Optionally link a GitLab repository and branch for artifact pushing
    - Returns session_id to track progress
    """

    # Create session
    session_id = str(uuid.uuid4())
    session_dir = create_session_directory(session_id)

    # Save all uploaded files
    file_paths = []
    filenames = []
    for file in files:
        file_path = session_dir / file.filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(str(file_path))
        filenames.append(file.filename)

    # Initialize state
    initial_state = {
        "task": task,
        "data_file_path": file_paths[0],  # First file for backward compatibility - agent can use this
        "data_file_paths": file_paths,  # All file paths - agent can use this to access all files
        "output_dir": str(session_dir),  # Full path for plots (used in generated code)
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

    # Parse require_code_approval (comes as string from form)
    require_code_approval_bool = require_code_approval.lower() == 'true'

    # Store session
    sessions[session_id] = {
        "session_id": session_id,
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "task": task,
        "data_file": filenames[0] if len(filenames) == 1 else f"{len(filenames)} files",  # For display
        "data_files": filenames,  # All filenames
        "state": initial_state,
        "session_dir": str(session_dir),
        "artifact_version": 0,  # Start at 0, will increment to 1 on first save
        "refinement_count": 0,
        "require_code_approval": require_code_approval_bool,  # Store per-session code approval setting
        "gitlab_repo": gitlab_repo,  # Store GitLab repo URL if provided
        "gitlab_branch": gitlab_branch if gitlab_repo else None  # Store GitLab branch if repo provided
    }

    # Start analysis in background
    asyncio.create_task(run_analysis_async(session_id, initial_state))

    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Analysis started. Use /analysis/{session_id}/stream to watch progress."
    }


async def run_analysis_async(session_id: str, state: StatisticalAnalysisState):
    """Run analysis asynchronously in the background."""
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_agent_with_streaming, session_id, state)
    except Exception as e:
        sessions[session_id]["status"] = "failed"
        sessions[session_id]["error"] = str(e)


@app.get("/analysis/{session_id}/status")
async def get_status(session_id: str):
    """Get the current status of an analysis session."""
    session = get_session(session_id)
    state = session["state"]

    response = {
        "session_id": session_id,
        "status": session["status"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "task": session["task"],
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 3),
        "is_valid": state.get("is_valid", False),
        "has_errors": bool(state.get("execution_error", "")),
        "reasoning_entries": len(state.get("reasoning_log", [])),
        "feedback_link": session.get("feedback_link"),  # Include feedback link if available
        "gitlab_repo": session.get("gitlab_repo"),  # Include GitLab repo if linked
        "gitlab_branch": session.get("gitlab_branch")  # Include GitLab branch if set
    }

    # Include pending code if awaiting approval
    if session["status"] == "awaiting_approval":
        response["pending_code"] = session.get("pending_code", "")

    return response


@app.get("/analysis/{session_id}/stream")
async def stream_reasoning_events(session_id: str):
    """
    Stream reasoning events in real-time using Server-Sent Events (SSE).

    Connect to this endpoint to receive live updates as the agent works.
    """
    session = get_session(session_id)

    async def event_generator():
        """Generate SSE events."""
        last_count = 0

        while True:
            # Check if session still exists
            if session_id not in sessions:
                break

            current_session = sessions[session_id]
            state = current_session["state"]
            reasoning_log = state.get("reasoning_log", [])

            # Check if completed
            is_completed = current_session["status"] in ["completed", "failed", "cancelled"]

            # Send new reasoning entries (including any final ones if completed)
            if len(reasoning_log) > last_count:
                for entry in reasoning_log[last_count:]:
                    event = {
                        "session_id": session_id,
                        "node": entry.get("node", "unknown"),
                        "iteration": entry.get("iteration", 0),
                        "thought": entry.get("thought", ""),
                        "timestamp": entry.get("timestamp", "")
                    }
                    yield f"data: {json.dumps(event)}\n\n"

                last_count = len(reasoning_log)

            # Send completion event after all reasoning entries
            if is_completed:
                completion_event = {
                    "type": "completion",
                    "session_id": session_id,
                    "status": current_session["status"],
                    "is_valid": state.get("is_valid", False)
                }
                yield f"data: {json.dumps(completion_event)}\n\n"
                break

            await asyncio.sleep(0.5)  # Poll every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/analysis/{session_id}/refine")
async def refine_analysis(session_id: str, request: RefineRequest):
    """
    Refine an existing analysis with a follow-up query.

    This keeps the existing state and data, but applies a new refinement prompt.
    """
    session = get_session(session_id)

    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Cannot refine analysis that is not completed"
        )

    # Get existing state
    previous_state = session["state"]

    # Create new task that includes context
    refined_task = f"""REFINEMENT REQUEST:
{request.refinement_prompt}

ORIGINAL TASK:
{previous_state.get('task', '')}

PREVIOUS ANALYSIS OBJECTIVE:
{previous_state.get('analysis_objective', '')}

PREVIOUS CONCLUSIONS:
{previous_state.get('analysis_conclusions', '')}

Please refine the analysis based on the refinement request above."""

    # Update state for refinement
    refinement_state = {
        **previous_state,
        "task": refined_task,
        "iteration": 0,  # Reset iteration counter
        "max_iterations": request.max_iterations,
        "should_continue": True,
        "is_valid": False,
        # Keep: data_file_path, data_summary, reasoning_log (to maintain history)
    }

    # Update session (preserve require_code_approval setting)
    session["status"] = "processing"
    session["updated_at"] = datetime.now().isoformat()
    session["state"] = refinement_state
    session["refinement_count"] = session.get("refinement_count", 0) + 1
    # Note: require_code_approval is preserved from the original session

    # Start refinement in background
    asyncio.create_task(run_analysis_async(session_id, refinement_state))

    return {
        "session_id": session_id,
        "status": "processing",
        "message": "Refinement started. Use /analysis/{session_id}/stream to watch progress.",
        "refinement_count": session["refinement_count"]
    }


@app.get("/analysis/{session_id}/files/{filename}")
async def download_file(session_id: str, filename: str):
    """
    Download any file from an analysis session.

    Supports:
    - Reports (.txt)
    - Code (.py)
    - Plots (.png, .jpg, .svg, .pdf)
    - Any other generated files
    """
    session = get_session(session_id)

    # Security: prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    session_dir = Path(session["session_dir"])
    file_path = session_dir / filename

    # Check file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Check file is actually in session directory (security)
    if session_dir not in file_path.parents and file_path.parent != session_dir:
        raise HTTPException(status_code=403, detail="Access denied")

    # Determine media type based on extension
    extension = file_path.suffix.lower()
    media_type_map = {
        '.txt': 'text/plain',
        '.py': 'text/plain',  # Changed to text/plain for better browser rendering
        '.yml': 'text/plain',  # YAML as plain text for browser display
        '.yaml': 'text/plain',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
    }
    media_type = media_type_map.get(extension, 'application/octet-stream')

    # For viewable files, don't set filename to allow inline display
    # Only force download for unknown file types
    if extension in ['.txt', '.py', '.yml', '.yaml', '.png', '.jpg', '.jpeg', '.svg', '.pdf']:
        return FileResponse(
            path=file_path,
            media_type=media_type
        )
    else:
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )


@app.get("/analysis/{session_id}/artifacts")
async def list_artifacts(session_id: str):
    """
    List all artifacts for a session, including versioned files and plots.

    Returns:
        - reports: list of report files
        - reasoning: list of reasoning files
        - code: list of code files
        - plots: list of plot/image files
    """
    session = get_session(session_id)
    session_dir = Path(session["session_dir"])

    if not session_dir.exists():
        return {"reports": [], "reasoning": [], "code": [], "plots": []}

    artifacts = {
        "reports": [],
        "reasoning": [],
        "code": [],
        "plots": []
    }

    # Scan directory for files
    for file_path in session_dir.iterdir():
        if file_path.is_file():
            filename = file_path.name

            if filename.startswith("analysis_report") and (filename.endswith(".txt") or filename.endswith(".yml")):
                artifacts["reports"].append({
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            elif filename.startswith("agent_reasoning") and filename.endswith(".txt"):
                artifacts["reasoning"].append({
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            elif filename.startswith("analysis_code") and filename.endswith(".py"):
                artifacts["code"].append({
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                artifacts["plots"].append({
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

    # Sort by modified time (newest first)
    for key in artifacts:
        artifacts[key].sort(key=lambda x: x["modified"], reverse=True)

    return artifacts


@app.post("/analysis/{session_id}/question")
async def ask_question(session_id: str, question: str = Form(...)):
    """
    Ask a question about the analysis without re-running.

    This uses the existing analysis results to answer questions.
    """
    session = get_session(session_id)

    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Cannot answer questions about incomplete analysis"
        )

    # Get existing results
    session_dir = Path(session["session_dir"])
    state = session.get("state", {})

    # Read latest report and code
    report_path = session_dir / "analysis_report.txt"
    code_path = session_dir / "analysis_code.py"

    context = f"""
Analysis Task: {state.get('task', 'N/A')}

Analysis Objective:
{state.get('analysis_objective', 'N/A')}

Analysis Details:
{state.get('analysis_details', 'N/A')}

Analysis Conclusions:
{state.get('analysis_conclusions', 'N/A')}
"""

    if code_path.exists():
        with open(code_path, 'r') as f:
            context += f"\n\nCode:\n{f.read()[:2000]}..."

    # Use LLM to answer question based on context
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    system_prompt = """You are a statistical analysis assistant. Answer questions about the analysis results based on the provided context.
Be specific and reference the actual results. If the information isn't in the context, say so."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Context:\n{context}\n\nQuestion: {question}""")
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id
    }


@app.post("/analysis/{session_id}/stop")
async def stop_analysis(session_id: str):
    """
    Stop/cancel a running analysis.

    This sets a flag that the agent will check to stop execution gracefully.
    """
    session = get_session(session_id)

    if session["status"] not in ["processing", "awaiting_approval"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot stop analysis in status: {session['status']}"
        )

    # Mark session as cancelled
    session["status"] = "cancelled"
    session["updated_at"] = datetime.now().isoformat()

    # Save metadata
    save_session_metadata()

    return {
        "message": f"Analysis {session_id} stopped",
        "status": "cancelled"
    }


@app.post("/analysis/{session_id}/approve_code")
async def approve_code(session_id: str):
    """
    Approve the generated code for execution.

    Only works when REQUIRE_CODE_APPROVAL is enabled and session is awaiting approval.
    """
    session = get_session(session_id)

    if session["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not awaiting approval (status: {session['status']})"
        )

    # Mark as approved
    session["approval_status"] = "approved"
    session["status"] = "processing"
    session["updated_at"] = datetime.now().isoformat()

    # Save metadata
    save_session_metadata()

    return {
        "message": "Code approved",
        "session_id": session_id
    }


@app.post("/analysis/{session_id}/reject_code")
async def reject_code(session_id: str):
    """
    Reject the generated code and stop the analysis.

    Only works when REQUIRE_CODE_APPROVAL is enabled and session is awaiting approval.
    """
    session = get_session(session_id)

    if session["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not awaiting approval (status: {session['status']})"
        )

    # Mark as rejected and stop
    session["approval_status"] = "rejected"
    session["status"] = "cancelled"
    session["updated_at"] = datetime.now().isoformat()

    # Save metadata
    save_session_metadata()

    return {
        "message": "Code rejected, analysis stopped",
        "session_id": session_id
    }


@app.post("/analysis/{session_id}/push-to-gitlab")
async def push_to_gitlab(session_id: str, request: GitLabPushRequest):
    """
    Push analysis artifacts to a GitLab repository.

    Requires GITLAB_TOKEN environment variable to be set.

    Args:
        session_id: The session ID
        request: GitLabPushRequest with gitlab_repo and gitlab_branch

    Example usage:
    export GITLAB_TOKEN=your_personal_access_token
    """
    session = get_session(session_id)

    # Get repo and branch from request
    gitlab_repo = request.gitlab_repo
    gitlab_branch = request.gitlab_branch

    if not gitlab_repo:
        raise HTTPException(
            status_code=400,
            detail="GitLab repository URL is required"
        )

    # Check if analysis is completed
    if session["status"] not in ["completed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot push artifacts from session with status: {session['status']}"
        )

    # SAMPLE IMPLEMENTATION - Uncomment and install python-gitlab to use
    r"""
    # 1. Get GitLab token from environment
    import gitlab
    gitlab_token = os.getenv('GITLAB_TOKEN')
    if not gitlab_token:
        raise HTTPException(
            status_code=500,
            detail="GITLAB_TOKEN environment variable not set"
        )

    # 2. Parse repo URL to extract project path
    # Example: https://gitlab.com/username/repo.git -> username/repo
    import re
    match = re.search(r'gitlab\.com[/:]([^/]+/[^/.]+)', gitlab_repo)
    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GitLab repo URL: {gitlab_repo}"
        )
    project_path = match.group(1)

    # 3. Connect to GitLab
    gl = gitlab.Gitlab('https://gitlab.com', private_token=gitlab_token)
    gl.auth()

    # 4. Get the project
    project = gl.projects.get(project_path)

    # 5. Get session directory with artifacts
    session_dir = Path(session["session_dir"])

    # 6. Prepare files to commit
    # Organize artifacts into different directories in the repo
    actions = []

    # Add reports to /reports directory
    for file in session_dir.glob("analysis_report*.yml"):
        with open(file, 'r') as f:
            content = f.read()
        actions.append({
            'action': 'create',
            'file_path': f'reports/{file.name}',
            'content': content
        })

    # Add reasoning logs to /reasoning directory
    for file in session_dir.glob("agent_reasoning*.txt"):
        with open(file, 'r') as f:
            content = f.read()
        actions.append({
            'action': 'create',
            'file_path': f'reasoning/{file.name}',
            'content': content
        })

    # Add code to /code directory
    for file in session_dir.glob("analysis_code*.py"):
        with open(file, 'r') as f:
            content = f.read()
        actions.append({
            'action': 'create',
            'file_path': f'code/{file.name}',
            'content': content
        })

    # Add plots to /plots directory (as binary)
    for file in session_dir.glob("*.png"):
        with open(file, 'rb') as f:
            import base64
            content = base64.b64encode(f.read()).decode('utf-8')
        actions.append({
            'action': 'create',
            'file_path': f'plots/{file.name}',
            'content': content,
            'encoding': 'base64'
        })

    # 7. Create commit with all artifacts
    commit_message = f'''Add analysis artifacts for session {session_id[:8]}

Session: {session_id}
Status: {session["status"]}
Created: {session["created_at"]}

Automated commit from Statistical Analysis Agent
'''

    data = {
        'branch': gitlab_branch,  # Use the branch from request
        'commit_message': commit_message,
        'actions': actions
    }

    # Make the commit
    commit = project.commits.create(data)

    return {
        "message": "Successfully pushed artifacts to GitLab",
        "session_id": session_id,
        "gitlab_repo": gitlab_repo,
        "gitlab_branch": gitlab_branch,
        "commit_sha": commit.id,
        "commit_url": commit.web_url,
        "files_pushed": len(actions)
    }
    """

    # For now, return a placeholder success message
    # To enable GitLab push:
    # 1. Install: pip install python-gitlab
    # 2. Set environment variable: export GITLAB_TOKEN=your_token
    # 3. Uncomment the code above
    return {
        "message": f"Placeholder: Would push artifacts to {gitlab_repo} (branch: {gitlab_branch})",
        "session_id": session_id,
        "gitlab_repo": gitlab_repo,
        "gitlab_branch": gitlab_branch,
        "status": "placeholder_success",
        "note": "GitLab push is not yet enabled. See code comments for implementation details."
    }


@app.post("/analysis/{session_id}/set_feedback_link")
async def set_feedback_link(session_id: str, feedback_url: str = Form(...)):
    """
    Set the feedback link for a session.

    This allows you to generate and set the feedback link asynchronously
    after analysis completion. Call this endpoint once your feedback link
    is ready to be displayed to the user.

    Args:
        session_id: The session ID
        feedback_url: The feedback URL to set for this session

    Returns:
        Confirmation with the set feedback link
    """
    session = get_session(session_id)

    # Set the feedback link
    session["feedback_link"] = feedback_url
    session["updated_at"] = datetime.now().isoformat()

    # Save metadata
    save_session_metadata()

    return {
        "message": f"Feedback link set for session {session_id}",
        "session_id": session_id,
        "feedback_link": feedback_url
    }


@app.delete("/analysis/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated files."""
    session = get_session(session_id)

    # Delete session directory
    session_dir = Path(session["session_dir"])
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # Remove from sessions
    del sessions[session_id]

    # Update metadata
    save_session_metadata()

    return {
        "message": f"Session {session_id} deleted successfully"
    }


@app.get("/sessions")
async def list_sessions():
    """List all sessions with details."""
    session_list = []
    for sid, session in sessions.items():
        session_list.append({
            "session_id": sid,
            "status": session["status"],
            "created_at": session["created_at"],
            "updated_at": session.get("updated_at", session["created_at"]),
            "task": session["task"][:100] + "..." if len(session["task"]) > 100 else session["task"],
            "data_file": session.get("data_file", ""),
            "refinement_count": session.get("refinement_count", 0)
        })

    # Sort by updated_at (most recent first)
    session_list.sort(key=lambda x: x["updated_at"], reverse=True)

    return {"sessions": session_list}


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("Statistical Analysis Agent API started")
    print(f"Sessions directory: {SESSIONS_DIR.absolute()}")

    # Load persisted sessions
    metadata = load_session_metadata()
    for sid, meta in metadata.items():
        # Restore session metadata (without full state)
        sessions[sid] = {
            **meta,
            "state": {}  # State will be loaded on demand
        }
    print(f"Loaded {len(sessions)} persisted session(s)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("Statistical Analysis Agent API")
    print("=" * 60)
    print("\n🌐 Web UI:        http://localhost:9002")
    print("📚 API Docs:      http://localhost:9002/docs")
    print("ℹ️  API Info:      http://localhost:9002/api/info")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=9002)
