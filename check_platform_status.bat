@echo off
cls
echo ========================================
echo 🔍 NEURADESK PLATFORM STATUS CHECK
echo ========================================
echo.

echo [1/5] Checking Backend Server...
curl -s http://localhost:8000/ >nul 2>&1
if %errorlevel%==0 (
    echo ✅ Backend is running on http://localhost:8000
) else (
    echo ❌ Backend is NOT running
    echo    Start it with: cd backend ^&^& python -m uvicorn main:app --reload --port 8000
)
echo.

echo [2/5] Checking Frontend Server...
curl -s http://localhost:5174/ >nul 2>&1
if %errorlevel%==0 (
    echo ✅ Frontend is running on http://localhost:5174
) else (
    curl -s http://localhost:5173/ >nul 2>&1
    if %errorlevel%==0 (
        echo ✅ Frontend is running on http://localhost:5173
    ) else (
        echo ❌ Frontend is NOT running
        echo    Start it with: npm run dev
    )
)
echo.

echo [3/5] Checking Database...
if exist "backend\neuradesk.db" (
    echo ✅ Database file exists: backend\neuradesk.db
    cd backend
    python -c "from database_sqlite import get_ticket_count; print(f'   Tickets in database: {get_ticket_count()}')" 2>nul
    cd ..
) else (
    echo ❌ Database file NOT found
    echo    It will be created when backend starts
)
echo.

echo [4/5] Checking Gemini API Key...
if exist "backend\.env" (
    findstr /C:"AIzaSy" backend\.env >nul 2>&1
    if %errorlevel%==0 (
        echo ✅ Gemini API key is configured
    ) else (
        echo ⚠️  Gemini API key might not be configured
        echo    Check backend\.env file
    )
) else (
    echo ❌ .env file NOT found
    echo    Create backend\.env and add your Gemini API key
)
echo.

echo [5/5] Platform URLs:
echo    📱 Customer Portal: http://localhost:5174/customer-portal
echo    📊 Dashboard: http://localhost:5174/
echo    🤖 AI Brain: http://localhost:5174/ai-brain
echo    🎯 Action Layer: http://localhost:5174/actions
echo    🎫 Tickets: http://localhost:5174/tickets
echo    🔧 Backend API: http://localhost:8000/api/tickets
echo.

echo ========================================
echo 📋 QUICK ACTIONS
echo ========================================
echo.
echo To view database: run view_database.bat
echo To manage database: cd backend ^&^& python db_manager.py
echo To test connection: open test_connection.html
echo.
pause
