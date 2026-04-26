@echo off
echo ========================================
echo Fake Food Image Detection Platform
echo Quick Start Script
echo ========================================
echo.

echo Step 1: Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
echo.

echo Step 2: Checking Node.js installation...
node --version
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js 16+ from nodejs.org
    pause
    exit /b 1
)
echo.

echo Step 3: Checking Gemini API key...
if not exist "backend\.env" (
    echo WARNING: backend\.env file not found!
    echo Creating from template...
    copy backend\.env.example backend\.env
    echo.
    echo IMPORTANT: Edit backend\.env and add your Gemini API key!
    echo Get it from: https://makersuite.google.com/app/apikey
    echo.
    pause
)
echo.

echo Step 4: Installing backend dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies!
    pause
    exit /b 1
)
cd ..
echo.

echo Step 5: Installing frontend dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies!
    pause
    exit /b 1
)
echo.

echo Step 6: Running tests...
cd backend
python test_fraud_detection.py
cd ..
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo.
echo Terminal 1 - Backend:
echo   cd backend
echo   python -m uvicorn main:app --reload --port 8000
echo.
echo Terminal 2 - Frontend:
echo   npm run dev
echo.
echo Then open: http://localhost:5173/customer-portal
echo.
pause
