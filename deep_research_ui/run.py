import subprocess
import os
import sys
from pathlib import Path
import threading


def install_requirements():
    backend_requirements = Path.cwd() / "requirements.txt"
    try:
        print("Installing Python requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r",
            str(backend_requirements)
        ],
                       check=True)
        print("Python requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {str(e)}")
        sys.exit(1)


def stream_output(process, prefix):
    """Helper function to stream process output in a separate thread"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[{prefix}] {line.strip()}")
    process.stdout.close()


def stream_error(process, prefix):
    """Helper function to stream process errors in a separate thread"""
    for line in iter(process.stderr.readline, ''):
        if line:
            print(f"[{prefix} Error] {line.strip()}", file=sys.stderr)
    process.stderr.close()


def run_processes():
    # Get the current directory and set up paths
    current_dir = Path.cwd()
    backend_path = current_dir / "backend" / "research_agents"
    frontend_path = current_dir / "frontend"

    # Install Python requirements first
    install_requirements()

    try:
        print("\nStarting frontend and backend processes...")

        # Start the frontend process
        frontend_process = subprocess.Popen("npm install && npm run dev",
                                            cwd=frontend_path,
                                            shell=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            bufsize=1)

        # Start the research agent process
        research_agent_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"],
            cwd=backend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1)

        # Create threads for handling output
        frontend_output = threading.Thread(target=stream_output,
                                           args=(frontend_process, "Frontend"))
        frontend_error = threading.Thread(target=stream_error,
                                          args=(frontend_process, "Frontend"))
        agent_output = threading.Thread(target=stream_output,
                                        args=(research_agent_process,
                                              "Research Agent"))
        agent_error = threading.Thread(target=stream_error,
                                       args=(research_agent_process,
                                             "Research Agent"))

        # Start all threads
        frontend_output.start()
        frontend_error.start()
        agent_output.start()
        agent_error.start()

        # Wait for processes to complete
        frontend_process.wait()
        research_agent_process.wait()

        # Check for errors
        if frontend_process.returncode != 0:
            print("Frontend process failed!")
        if research_agent_process.returncode != 0:
            print("Research Agent process failed!")

    except KeyboardInterrupt:
        print("\nShutting down processes...")
        frontend_process.terminate()
        research_agent_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        # Make sure processes are terminated
        try:
            frontend_process.terminate()
            research_agent_process.terminate()
        except:
            pass


if __name__ == "__main__":
    run_processes()



