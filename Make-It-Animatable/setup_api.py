#!/usr/bin/env python3
"""
Setup script for Make-It-Animatable API server
Creates isolated environment to avoid ComfyUI conflicts
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run command and handle errors"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if check:
            raise
        return e

def setup_api_environment():
    """Setup isolated Python environment for API server"""
    
    # Paths
    base_dir = Path(__file__).parent
    api_server_dir = base_dir / "api_server"
    venv_dir = api_server_dir / ".venv"
    requirements_file = api_server_dir / "requirements_api.txt"
    
    print("🔧 Setting up Make-It-Animatable API environment...")
    print(f"Base directory: {base_dir}")
    print(f"API server directory: {api_server_dir}")
    
    # Create API server directory if it doesn't exist
    api_server_dir.mkdir(exist_ok=True)
    
    # Create virtual environment
    if venv_dir.exists():
        print("🔄 Virtual environment already exists, removing...")
        import shutil
        shutil.rmtree(venv_dir)
    
    print("🆕 Creating virtual environment...")
    venv.create(venv_dir, with_pip=True)
    
    # Determine Python executable path
    if os.name == "nt":  # Windows
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix-like
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    # Upgrade pip
    print("📦 Upgrading pip...")
    run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    if requirements_file.exists():
        print("📋 Installing requirements...")
        run_command([str(pip_exe), "install", "-r", str(requirements_file)])
    else:
        print("⚠️ Requirements file not found, installing basic dependencies...")
        run_command([str(pip_exe), "install", "fastapi", "uvicorn[standard]", "python-multipart", "requests"])
    
    # Create symbolic links to core Make-It-Animatable files (avoid duplication)
    print("🔗 Setting up symbolic links to Make-It-Animatable core files...")
    core_files = [
        "model.py", "models_ae.py", "engine.py",
        "util/", "data/", "output/"
    ]
    
    for file_path in core_files:
        src = base_dir / file_path
        dst = api_server_dir / file_path
        
        if src.exists():
            # Remove existing file/link if exists
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    import shutil
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            
            # Create symbolic link
            dst.symlink_to(src.resolve())
            print(f"🔗 Linked: {file_path} -> {src}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    # Create startup script
    startup_script = api_server_dir / "start_server.py"
    startup_script.write_text(f'''#!/usr/bin/env python3
"""
Startup script for Make-It-Animatable API server
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Change to API server directory
    api_dir = Path(__file__).parent
    os.chdir(api_dir)
    
    # Python executable in venv
    python_exe = api_dir / ".venv" / {"Scripts/python.exe" if os.name == "nt" else "bin/python"}
    
    # Start the server
    cmd = [str(python_exe), "main.py"]
    print(f"Starting API server: {{' '.join(cmd)}}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
''')
    
    print("✅ Setup complete!")
    print(f"📍 API server location: {api_server_dir}")
    print(f"🐍 Python executable: {python_exe}")
    print(f"🚀 To start server: python {startup_script}")
    print(f"🌐 API will be available at: http://127.0.0.1:8765")
    
    return True

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    api_server_dir = Path(__file__).parent / "api_server"
    python_exe = api_server_dir / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    
    # Test import
    test_cmd = [str(python_exe), "-c", "import torch; import fastapi; print('✅ Core imports successful')"]
    result = run_command(test_cmd, check=False)
    
    if result.returncode == 0:
        print("✅ Installation test passed!")
        return True
    else:
        print("❌ Installation test failed!")
        return False

def main():
    """Main setup function"""
    try:
        if setup_api_environment():
            if test_installation():
                print("\n🎉 Make-It-Animatable API setup completed successfully!")
                print("\nNext steps:")
                print("1. Start ComfyUI")
                print("2. Use the Make-It-Animatable nodes in your workflow")
                print("3. The API server will start automatically when needed")
            else:
                print("\n❌ Setup completed but tests failed. Check the installation.")
                return 1
        else:
            print("\n❌ Setup failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())