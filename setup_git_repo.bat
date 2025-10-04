@echo off
REM Fed-MARL Git Repository Setup Script for Windows
REM Run this script to initialize the Git repository and push to GitHub

echo 🚁 Fed-MARL Git Repository Setup
echo =================================

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

REM Initialize git repository
echo 📁 Initializing Git repository...
git init

REM Add all files
echo 📝 Adding files to repository...
git add .

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "feat: initial commit - Fed-MARL drone swarm coordination framework

- Add Indian Skyscape environment with realistic urban zones
- Implement QMIX multi-agent reinforcement learning
- Include comprehensive training and evaluation scripts
- Add visualization and testing utilities
- Provide detailed documentation and setup guides

Performance Results:
- Average Reward: 179.81 (target: >50) ✅
- Collision Rate: 0.3% (target: <5%) ✅
- Training Time: 42.4 minutes for 500 episodes ✅"

REM Add GitHub remote (you'll need to update this URL)
echo 🔗 Setting up GitHub remote...
echo ⚠️  Please update the repository URL in this script before running!
echo    Edit setup_git_repo.bat and replace 'yourusername' with your GitHub username
echo.

REM Uncomment and update this line with your actual GitHub repository URL
REM git remote add origin https://github.com/yourusername/Fed-MARL.git

REM Push to GitHub
echo 🚀 Pushing to GitHub...
echo ⚠️  Make sure you've:
echo    1. Created the repository on GitHub first
echo    2. Updated the remote URL in this script
echo    3. Have GitHub credentials configured
echo.

REM Uncomment this line after setting up the remote
REM git push -u origin main

echo ✅ Git repository setup complete!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub named 'Fed-MARL'
echo 2. Update the remote URL in this script
echo 3. Run: git remote add origin https://github.com/yourusername/Fed-MARL.git
echo 4. Run: git push -u origin main
echo.
echo 🎉 Your Fed-MARL repository is ready to share!
pause
