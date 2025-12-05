import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


# --------------------------------------------------------------
# RIVION PROJECT STRUCTURE
# --------------------------------------------------------------
list_of_files = [

    # Root
    "RIVION/README.md",
    "RIVION/CONTRIBUTING.md",
    "RIVION/LICENSE",
    "RIVION/ARCHITECTURE.md",
    "RIVION/.gitignore",
    "RIVION/docker-compose.yml",
    "RIVION/docker-compose.prod.yml",

    # Frontend root
    "RIVION/frontend/package.json",
    "RIVION/frontend/tsconfig.json",
    "RIVION/frontend/vite.config.ts",
    "RIVION/frontend/tailwind.config.js",
    "RIVION/frontend/.env.example",
    "RIVION/frontend/.gitignore",

    # Frontend public
    "RIVION/frontend/public/index.html",
    "RIVION/frontend/public/favicon.ico",

    # Frontend src core
    "RIVION/frontend/src/App.tsx",
    "RIVION/frontend/src/main.tsx",
    "RIVION/frontend/src/index.css",

    # Frontend components
    "RIVION/frontend/src/components/VideoChat/VideoChat.tsx",
    "RIVION/frontend/src/components/VideoChat/EmotionDisplay.tsx",
    "RIVION/frontend/src/components/VideoChat/ChatBox.tsx",
    "RIVION/frontend/src/components/VideoChat/AvatarDisplay.tsx",

    "RIVION/frontend/src/components/Dashboard/Dashboard.tsx",
    "RIVION/frontend/src/components/Dashboard/StressGauge.tsx",
    "RIVION/frontend/src/components/Dashboard/EmotionTimeline.tsx",
    "RIVION/frontend/src/components/Dashboard/HistoricalTrends.tsx",
    "RIVION/frontend/src/components/Dashboard/EmotionDistribution.tsx",

    "RIVION/frontend/src/components/Common/Header.tsx",
    "RIVION/frontend/src/components/Common/Sidebar.tsx",
    "RIVION/frontend/src/components/Common/Modal.tsx",
    "RIVION/frontend/src/components/Common/Button.tsx",

    "RIVION/frontend/src/components/Auth/Login.tsx",
    "RIVION/frontend/src/components/Auth/Register.tsx",
    "RIVION/frontend/src/components/Auth/ProtectedRoute.tsx",

    # Frontend pages
    "RIVION/frontend/src/pages/HomePage.tsx",
    "RIVION/frontend/src/pages/ChatPage.tsx",
    "RIVION/frontend/src/pages/DashboardPage.tsx",
    "RIVION/frontend/src/pages/SettingsPage.tsx",

    # Frontend hooks
    "RIVION/frontend/src/hooks/useWebRTC.ts",
    "RIVION/frontend/src/hooks/useEmotion.ts",
    "RIVION/frontend/src/hooks/useLocalStorage.ts",
    "RIVION/frontend/src/hooks/useAuth.ts",

    # Frontend store
    "RIVION/frontend/src/store/authStore.ts",
    "RIVION/frontend/src/store/chatStore.ts",
    "RIVION/frontend/src/store/emotionStore.ts",

    # Frontend services
    "RIVION/frontend/src/services/api.ts",
    "RIVION/frontend/src/services/authService.ts",
    "RIVION/frontend/src/services/chatService.ts",
    "RIVION/frontend/src/services/emotionService.ts",
    "RIVION/frontend/src/services/dashboardService.ts",

    # Frontend types
    "RIVION/frontend/src/types/emotion.ts",
    "RIVION/frontend/src/types/api.ts",
    "RIVION/frontend/src/types/user.ts",
    "RIVION/frontend/src/types/chat.ts",

    # Frontend utils
    "RIVION/frontend/src/utils/formatters.ts",
    "RIVION/frontend/src/utils/validators.ts",
    "RIVION/frontend/src/utils/constants.ts",
    "RIVION/frontend/src/utils/logger.ts",

    # Frontend styles
    "RIVION/frontend/src/styles/globals.css",
    "RIVION/frontend/src/styles/variables.css",
    "RIVION/frontend/src/styles/animations.css",

    # Backend root
    "RIVION/backend/requirements.txt",
    "RIVION/backend/requirements-dev.txt",
    "RIVION/backend/Dockerfile",
    "RIVION/backend/.dockerignore",
    "RIVION/backend/.env.example",
    "RIVION/backend/pyproject.toml",

    # Backend app core
    "RIVION/backend/app/__init__.py",
    "RIVION/backend/app/main.py",
    "RIVION/backend/app/config.py",
    "RIVION/backend/app/dependencies.py",
    "RIVION/backend/app/middleware.py",

    # Backend routers
    "RIVION/backend/app/routers/__init__.py",
    "RIVION/backend/app/routers/auth.py",
    "RIVION/backend/app/routers/chat.py",
    "RIVION/backend/app/routers/emotion.py",
    "RIVION/backend/app/routers/dashboard.py",
    "RIVION/backend/app/routers/health.py",

    # Backend models
    "RIVION/backend/app/models/__init__.py",
    "RIVION/backend/app/models/base.py",
    "RIVION/backend/app/models/user.py",
    "RIVION/backend/app/models/session.py",
    "RIVION/backend/app/models/emotion_log.py",
    "RIVION/backend/app/models/message.py",

    # Backend schemas
    "RIVION/backend/app/schemas/__init__.py",
    "RIVION/backend/app/schemas/user.py",
    "RIVION/backend/app/schemas/emotion.py",
    "RIVION/backend/app/schemas/chat.py",
    "RIVION/backend/app/schemas/auth.py",
    "RIVION/backend/app/schemas/dashboard.py",

    # Backend services
    "RIVION/backend/app/services/__init__.py",
    "RIVION/backend/app/services/face_emotion_service.py",
    "RIVION/backend/app/services/voice_emotion_service.py",
    "RIVION/backend/app/services/text_emotion_service.py",
    "RIVION/backend/app/services/fusion_service.py",
    "RIVION/backend/app/services/llm_service.py",
    "RIVION/backend/app/services/auth_service.py",
    "RIVION/backend/app/services/chat_service.py",
    "RIVION/backend/app/services/emotion_service.py",
    "RIVION/backend/app/services/dashboard_service.py",

    # Backend utils
    "RIVION/backend/app/utils/__init__.py",
    "RIVION/backend/app/utils/logger.py",
    "RIVION/backend/app/utils/validators.py",
    "RIVION/backend/app/utils/decorators.py",
    "RIVION/backend/app/utils/constants.py",

    # Backend DB
    "RIVION/backend/app/db/__init__.py",
    "RIVION/backend/app/db/database.py",
    "RIVION/backend/app/db/session.py",
    "RIVION/backend/app/db/init_db.py",

    # Backend ML models folders
    "RIVION/backend/app/ml_models/__init__.py",
    "RIVION/backend/app/ml_models/face_models/.gitkeep",
    "RIVION/backend/app/ml_models/voice_models/.gitkeep",
    "RIVION/backend/app/ml_models/text_models/.gitkeep",

    # Backend WebSocket
    "RIVION/backend/app/websocket/__init__.py",
    "RIVION/backend/app/websocket/connection_manager.py",
    "RIVION/backend/app/websocket/event_handlers.py",
    "RIVION/backend/app/websocket/middleware.py",

    # Backend migrations & tests
    "RIVION/backend/migrations/.gitkeep",
    "RIVION/backend/tests/__init__.py",
    "RIVION/backend/tests/conftest.py",
    "RIVION/backend/tests/unit/.gitkeep",
    "RIVION/backend/tests/integration/.gitkeep",
    "RIVION/backend/tests/e2e/.gitkeep",

    # Optional dashboard app
    "RIVION/dashboard/.gitkeep",

    # GitHub Actions
    "RIVION/.github/workflows/test.yml",
    "RIVION/.github/workflows/build.yml",
    "RIVION/.github/workflows/deploy.yml",
]


# --------------------------------------------------------------
# CREATE FOLDERS + FILES
# --------------------------------------------------------------
for filepath in list_of_files:
    path = Path(filepath)
    directory, filename = os.path.split(path)

    # Create directory if needed
    if directory != "":
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

    # Create file if it does not exist or is empty
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        with open(path, "w", encoding="utf-8") as f:
            # Optional: write tiny hints into important files
            if filename in {"main.py", "App.tsx", "Dashboard.tsx"}:
                f.write("# TODO: implement\n")
        logging.info(f"Created file: {path}")
    else:
        logging.info(f"File exists: {path}")

logging.info("RIVION project structure creation complete.")
