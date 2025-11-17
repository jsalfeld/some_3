"""
Example client for the Statistical Analysis Agent API

Demonstrates:
1. Creating a new analysis (with optional GitLab repo linking)
2. Polling for completion status
3. Streaming reasoning steps in real-time
4. Retrieving the analysis report (YAML format)
5. Refining the analysis with follow-up queries
6. Downloading all artifacts
7. Pushing artifacts to GitLab

Usage:
    python api_client_example.py

Make sure the API server is running:
    python api.py
"""

import requests
import json
import time
import yaml
from pathlib import Path
from typing import Optional


API_BASE_URL = "http://localhost:9002"


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def create_analysis(
    data_file_path: str,
    task: str,
    max_iterations: int = 3,
    gitlab_repo: Optional[str] = None
):
    """
    Create a new analysis session.

    Args:
        data_file_path: Path to the data file (CSV, Excel, etc.)
        task: Description of the analysis task
        max_iterations: Maximum number of iterations for the agent
        gitlab_repo: Optional GitLab repository URL to link

    Returns:
        session_id if successful, None otherwise
    """
    print(f"Creating new analysis session...")
    print(f"Task: {task}")
    print(f"Data file: {data_file_path}")
    if gitlab_repo:
        print(f"GitLab repo: {gitlab_repo}")

    with open(data_file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'task': task,
            'max_iterations': max_iterations
        }
        if gitlab_repo:
            data['gitlab_repo'] = gitlab_repo

        response = requests.post(f"{API_BASE_URL}/analysis/new", files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Session created: {result['session_id']}")
        return result['session_id']
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return None


def stream_reasoning(session_id: str):
    """Stream reasoning events in real-time."""
    print(f"\nStreaming reasoning events for session {session_id}...")
    print_separator()

    url = f"{API_BASE_URL}/analysis/{session_id}/stream"

    try:
        with requests.get(url, stream=True, timeout=300) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])

                        # Check if it's a completion event
                        if data.get('type') == 'completion':
                            print_separator()
                            print(f"✓ Analysis {data['status']}")
                            print(f"Valid: {data['is_valid']}")
                            break

                        # Print reasoning event
                        node = data.get('node', 'unknown')
                        iteration = data.get('iteration', 0)
                        thought = data.get('thought', '')
                        timestamp = data.get('timestamp', '')

                        print(f"[{node}] (Iteration {iteration}) - {timestamp}")
                        print(thought)
                        print("-" * 80)

    except requests.exceptions.Timeout:
        print("Stream timed out")
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


def get_status(session_id: str):
    """Get the current status of an analysis."""
    response = requests.get(f"{API_BASE_URL}/analysis/{session_id}/status")

    if response.status_code == 200:
        status = response.json()
        print(f"\nSession Status:")
        print(f"  Status: {status['status']}")
        print(f"  Iteration: {status['iteration']}/{status['max_iterations']}")
        print(f"  Valid: {status['is_valid']}")
        print(f"  Reasoning entries: {status['reasoning_entries']}")
        return status
    else:
        print(f"Error: {response.status_code}")
        return None


def refine_analysis(session_id: str, refinement_prompt: str, max_iterations: int = 2):
    """Refine an existing analysis."""
    print(f"\nRefining analysis...")
    print(f"Refinement: {refinement_prompt}")

    data = {
        'refinement_prompt': refinement_prompt,
        'max_iterations': max_iterations
    }

    response = requests.post(f"{API_BASE_URL}/analysis/{session_id}/refine", json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Refinement started")
        return True
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return False


def download_file(session_id: str, filename: str, output_dir: str = "."):
    """Download a result file."""
    response = requests.get(f"{API_BASE_URL}/analysis/{session_id}/files/{filename}")

    if response.status_code == 200:
        output_path = Path(output_dir) / f"{session_id}_{filename}"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded: {output_path}")
        return output_path
    else:
        print(f"✗ Error downloading {filename}: {response.status_code}")
        return None


def download_all_files(session_id: str, output_dir: str = "."):
    """Download all result files from a session."""
    print(f"\nDownloading result files...")
    files = ["analysis_report.yml", "agent_reasoning.txt", "analysis_code.py"]

    for filename in files:
        download_file(session_id, filename, output_dir)


def poll_for_completion(session_id: str, poll_interval: int = 2, timeout: int = 300):
    """
    Poll the API until analysis completes.

    Args:
        session_id: The session ID to poll
        poll_interval: Seconds between polls
        timeout: Maximum time to wait in seconds

    Returns:
        Final status dict if completed, None if timeout
    """
    print(f"\nPolling for completion (timeout: {timeout}s)...")
    start_time = time.time()

    while True:
        status = get_status(session_id)

        if not status:
            print("✗ Failed to get status")
            return None

        if status['status'] in ['completed', 'cancelled', 'failed']:
            print(f"\n✓ Analysis {status['status']}")
            return status

        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n✗ Timeout after {elapsed:.0f}s")
            return None

        time.sleep(poll_interval)


def get_report(session_id: str) -> Optional[dict]:
    """
    Retrieve and parse the analysis report (YAML format).

    Returns:
        Parsed report as dictionary, or None if error
    """
    print(f"\nRetrieving analysis report...")

    response = requests.get(f"{API_BASE_URL}/analysis/{session_id}/files/analysis_report.yml")

    if response.status_code == 200:
        report = yaml.safe_load(response.text)
        print(f"✓ Report retrieved successfully")
        return report
    else:
        print(f"✗ Error retrieving report: {response.status_code}")
        return None


def print_report(report: dict):
    """Print a formatted analysis report."""
    if not report:
        return

    print_separator()
    print("ANALYSIS REPORT")
    print_separator()

    print(f"Title: {report.get('analysis_title', 'N/A')}")
    print(f"Label: {report.get('analysis_label', 'N/A')}")

    print(f"\nObjective:\n{report.get('analysis_objective', 'N/A')}")

    print(f"\nDetails:\n{report.get('analysis_details', 'N/A')}")

    print(f"\nConclusions:\n{report.get('analysis_conclusions', 'N/A')}")
    print_separator()


def list_artifacts(session_id: str):
    """List all artifacts for a session."""
    print(f"\nListing artifacts for session {session_id}...")

    response = requests.get(f"{API_BASE_URL}/analysis/{session_id}/artifacts")

    if response.status_code == 200:
        artifacts = response.json()

        print("\nReports:")
        for file in artifacts.get('reports', []):
            print(f"  - {file['filename']} ({file['size']} bytes)")

        print("\nReasoning Logs:")
        for file in artifacts.get('reasoning', []):
            print(f"  - {file['filename']} ({file['size']} bytes)")

        print("\nCode:")
        for file in artifacts.get('code', []):
            print(f"  - {file['filename']} ({file['size']} bytes)")

        print("\nPlots:")
        for file in artifacts.get('plots', []):
            print(f"  - {file['filename']} ({file['size']} bytes)")

        return artifacts
    else:
        print(f"✗ Error listing artifacts: {response.status_code}")
        return None


def push_to_gitlab(session_id: str):
    """Push analysis artifacts to linked GitLab repository."""
    print(f"\nPushing artifacts to GitLab...")

    response = requests.post(f"{API_BASE_URL}/analysis/{session_id}/push-to-gitlab")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ {result.get('message', 'Success')}")
        if 'commit_url' in result:
            print(f"  Commit URL: {result['commit_url']}")
            print(f"  Files pushed: {result.get('files_pushed', 0)}")
        return True
    else:
        error = response.json()
        print(f"✗ Error: {error.get('detail', 'Unknown error')}")
        return False


# ============================================================================
# Example Usage
# ============================================================================

def example_basic_analysis():
    """Example: Basic analysis workflow."""
    print_separator()
    print("EXAMPLE 1: Basic Analysis")
    print_separator()

    # Create sample data
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    returns = np.random.normal(0.001, 0.02, 500)
    df = pd.DataFrame({'date': dates, 'returns': returns})
    data_file = 'sample_data.csv'
    df.to_csv(data_file, index=False)
    print(f"Created sample data: {data_file}")

    # Create analysis
    task = "Calculate the 95% and 99% Value at Risk using historical method. Show the distribution of returns."
    session_id = create_analysis(data_file, task, max_iterations=3)

    if not session_id:
        return

    # Stream reasoning in real-time
    stream_reasoning(session_id)

    # Get final status
    get_status(session_id)

    # Download files
    download_all_files(session_id)

    print_separator()
    print("✓ Example 1 complete!")
    print(f"Session ID: {session_id}")

    return session_id


def example_refinement(session_id: str):
    """Example: Refining an existing analysis."""
    print_separator()
    print("EXAMPLE 2: Refining Analysis")
    print_separator()

    # Wait a moment to ensure first analysis is done
    time.sleep(2)

    # Refine the analysis
    refinement = "Now also calculate Expected Shortfall (CVaR) and compare it to VaR. Add a plot showing both metrics over time."

    if refine_analysis(session_id, refinement, max_iterations=2):
        # Stream the refinement reasoning
        stream_reasoning(session_id)

        # Get updated status
        get_status(session_id)

        # Download updated files
        download_all_files(session_id, output_dir="refined")

    print_separator()
    print("✓ Example 2 complete!")


def example_list_sessions():
    """Example: List all sessions."""
    print_separator()
    print("EXAMPLE 3: List Sessions")
    print_separator()

    response = requests.get(f"{API_BASE_URL}/sessions")

    if response.status_code == 200:
        sessions = response.json()['sessions']
        print(f"Found {len(sessions)} session(s):")
        for session in sessions:
            print(f"  - {session['session_id'][:8]}... | {session['status']} | {session['task'][:50]}")
    else:
        print(f"Error: {response.status_code}")


def example_polling_workflow():
    """Example: Poll-based workflow (no streaming)."""
    print_separator()
    print("EXAMPLE 4: Polling Workflow")
    print_separator()

    # Create sample data
    import pandas as pd
    import numpy as np

    np.random.seed(123)
    df = pd.DataFrame({
        'x': np.random.randn(200),
        'y': np.random.randn(200) * 2 + 5
    })
    data_file = 'sample_xy.csv'
    df.to_csv(data_file, index=False)
    print(f"Created sample data: {data_file}")

    # Create analysis
    task = "Test if the means of x and y are significantly different using a t-test."
    session_id = create_analysis(data_file, task, max_iterations=2)

    if not session_id:
        return None

    # Poll for completion instead of streaming
    final_status = poll_for_completion(session_id, poll_interval=3, timeout=180)

    if final_status:
        # List all artifacts
        list_artifacts(session_id)

        # Retrieve and print the report
        report = get_report(session_id)
        if report:
            print_report(report)

    print_separator()
    print("✓ Example 4 complete!")

    return session_id


def example_gitlab_integration():
    """Example: Analysis with GitLab integration."""
    print_separator()
    print("EXAMPLE 5: GitLab Integration")
    print_separator()

    # Create sample data
    import pandas as pd
    import numpy as np

    np.random.seed(456)
    df = pd.DataFrame({
        'returns': np.random.normal(0.0005, 0.015, 300)
    })
    data_file = 'sample_returns.csv'
    df.to_csv(data_file, index=False)
    print(f"Created sample data: {data_file}")

    # Create analysis WITH GitLab repo link
    task = "Calculate Sharpe ratio and maximum drawdown for the returns series."
    gitlab_repo = "https://gitlab.com/your-username/your-repo.git"  # Replace with real repo

    session_id = create_analysis(
        data_file,
        task,
        max_iterations=2,
        gitlab_repo=gitlab_repo
    )

    if not session_id:
        return None

    # Poll for completion
    final_status = poll_for_completion(session_id, poll_interval=3, timeout=180)

    if final_status and final_status['status'] == 'completed':
        # Get the report
        report = get_report(session_id)
        if report:
            print_report(report)

        # Push to GitLab
        print("\nAttempting to push artifacts to GitLab...")
        push_to_gitlab(session_id)

    print_separator()
    print("✓ Example 5 complete!")
    print("Note: GitLab push requires python-gitlab and GITLAB_TOKEN env var")

    return session_id


if __name__ == "__main__":
    print("Statistical Analysis Agent API - Client Example")
    print("Make sure the API is running: python api.py")
    print_separator()

    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print("✓ API is running")
    except requests.exceptions.ConnectionError:
        print("✗ API is not running. Please start it first with: python api.py")
        exit(1)

    # Run examples
    try:
        # Example 1: Basic analysis with streaming
        session_id = example_basic_analysis()

        if session_id:
            # Example 2: Refine the analysis
            example_refinement(session_id)

        # Example 3: List all sessions
        example_list_sessions()

        # Example 4: Polling-based workflow
        example_polling_workflow()

        # Example 5: GitLab integration (commented out by default)
        # Uncomment to test GitLab integration:
        # example_gitlab_integration()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print_separator()
    print("Examples complete!")
    print(f"Check your local directory for downloaded files.")
