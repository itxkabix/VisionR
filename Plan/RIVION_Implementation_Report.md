# 🚀 RIVION: Complete Implementation Report
## Real-Time Multimodal Emotion-Aware Conversational Agent

**Project**: RIVION (RoboCompanion alternative: Video-Call Based Mental Wellness Chatbot)  
**Status**: Comprehensive Technical & Implementation Plan  
**Date**: December 5, 2025  
**MCA Focus**: Full-Stack ML + Computer Vision + Audio + NLP + Production Deployment  

---

# TABLE OF CONTENTS

1. [Executive Overview](#executive-overview)
2. [Detailed Requirements Analysis](#detailed-requirements)
3. [Technology Stack (Complete)](#technology-stack)
4. [File Structure & Architecture](#file-structure)
5. [Database Design](#database-design)
6. [API Endpoints Specification](#api-endpoints)
7. [Phase-by-Phase Implementation](#implementation-phases)
8. [Development Environment Setup](#environment-setup)
9. [Deployment & DevOps](#deployment)
10. [Testing Strategy](#testing)
11. [Best Practices & Standards](#best-practices)
12. [Timeline & Resource Allocation](#timeline)
13. [Troubleshooting Guide](#troubleshooting)
14. [Appendix: Code Templates](#appendix)

---

# 1. EXECUTIVE OVERVIEW

## Project Vision
RIVION is a web-based mental wellness platform that detects user emotions in real-time through:
- **Visual input** (webcam): Face expressions, micro-expressions, arousal level
- **Audio input** (microphone): Tone, stress indicators, speech patterns
- **Text input** (chat): Sentiment, emotion keywords, explicit statements
- **Fusion engine**: Combines all 3 signals into reliable emotion score
- **Adaptive response**: LLM generates emotion-aware, supportive responses
- **Tracking**: Multi-session stress/burnout prediction

## Key Differentiators
✅ Real-time video interface (not batch processing)  
✅ Multimodal (3 signals, not 1-2)  
✅ Stress scoring (0-100, not just labels)  
✅ Burnout prediction (multi-session tracking)  
✅ Adaptive conversation (emotion-aware prompts)  
✅ Privacy-first (on-device face processing)  

---

# 2. DETAILED REQUIREMENTS ANALYSIS

## 2.1 Functional Requirements (What System Must Do)

### Core Features

#### A. Video Chat Interface
- **Requirement**: Real-time bidirectional video/audio streaming
- **Details**:
  - Video display: User's own webcam feed + optional AI avatar
  - Resolution: 720p minimum (1080p preferred)
  - FPS: 25-30 FPS minimum for smooth interaction
  - Latency: <300ms end-to-end (user sees/hears response quickly)
  - Audio: Mono or stereo, 16kHz sampling rate
  - Browser compatibility: Chrome, Firefox, Safari, Edge

#### B. Face Emotion Detection
- **Requirement**: Real-time facial expression classification
- **Details**:
  - 7-emotion classification: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral
  - Confidence score: 0-1 (min 0.6 confidence to count)
  - Processing: <100ms per frame (30 FPS = 33ms frame, need <100ms processing)
  - Multiple faces: If >1 face in frame, detect primary face (closest/largest)
  - Robustness: Handles lighting variations, head poses, glasses, masks
  - Arousal detection: Pupil dilation, eye opening, jaw tension
  - Action units: Track subtle micro-expressions

#### C. Voice Emotion Detection
- **Requirement**: Audio stream emotion classification + stress indicators
- **Details**:
  - Audio segments: 2-5 second chunks
  - Features extracted: MFCC (40 coefficients), pitch, energy, zero-crossing rate
  - Emotion classes: 7-class (same as face)
  - Stress indicators: Elevated pitch, irregular breathing, fast speech rate
  - Processing: <500ms per segment
  - Noise robustness: Filter background noise (utility noise, traffic, etc.)

#### D. Text Emotion Detection
- **Requirement**: Chat message emotion classification + keyword extraction
- **Details**:
  - Input: User text messages
  - Output: Emotion label + confidence + sentiment score (-1 to +1)
  - Keyword extraction: Risk indicators ("exhausted", "give up", "hopeless", "alone")
  - Sarcasm detection: "Oh great, another meeting" = negative (not positive)
  - Entity recognition: Extract what user is stressed about (project, relationship, health)
  - Response time: <50ms per message

#### E. Multimodal Fusion Engine
- **Requirement**: Combine face + voice + text into single emotion signal
- **Details**:
  - Input: 3 emotion predictions with confidence scores
  - Method: Attention-based weighted voting
  - Output: Final emotion label + stress score (0-100) + confidence
  - Hidden stress detection: Flag contradictions (smile + flat voice + negative text)
  - Temporal smoothing: Apply 10-frame moving average (eliminates flicker)

#### F. Adaptive Conversation Engine
- **Requirement**: Generate emotion-aware responses via LLM
- **Details**:
  - System prompt: Dynamically modified based on detected emotion
  - If stressed: Slower, softer, more supportive tone
  - If calm: More energetic, motivational tone
  - Safety: Detect crisis language → provide resources immediately
  - Context: Include emotion data in LLM prompt
  - Response time: <2 seconds (OpenAI API)

#### G. Stress Scoring & Burnout Prediction
- **Requirement**: Track stress over time + predict burnout risk
- **Details**:
  - Stress score: 0-100 (composite of arousal, vocal stress, text negativity)
  - Session tracking: Store score + emotion + keywords per session
  - Trend analysis: Compare current session to previous 5 sessions
  - Burnout risk: Low (<40), Medium (40-70), High (70-100)
  - Multi-session alert: If stress trend increasing → recommend professional help

#### H. Analytics Dashboard
- **Requirement**: Visualize emotion data + trends
- **Details**:
  - Real-time stress gauge (0-100 dial)
  - Session timeline: Mood variation during conversation
  - Historical trends: Stress level over days/weeks
  - Emotion distribution: Pie chart of emotions encountered
  - Intervention effectiveness: Did suggestions help? (user feedback)

---

## 2.2 Non-Functional Requirements (How System Must Work)

### Performance Requirements
| Metric | Target | Justification |
|--------|--------|---------------|
| **Face detection latency** | <100ms | 30 FPS video = 33ms/frame, need headroom for processing |
| **Voice inference** | <500ms per 5-sec segment | Real-time doesn't mean instant for audio |
| **Text emotion** | <50ms | Instant feedback on chat messages |
| **LLM response** | <2 seconds | User expects near-instant responses |
| **Overall end-to-end** | <1 second | For smooth conversational feel |
| **Video FPS** | 25-30 FPS | Smooth video, not choppy |
| **Concurrent users** | 100+ | FastAPI async can handle easily |

### Security Requirements
| Aspect | Requirement | Implementation |
|--------|-------------|-----------------|
| **Video data** | Never stored on server | Process on-device only (client-side) |
| **Audio clips** | Delete after 5 seconds | Only extract features, discard waveform |
| **Chat messages** | Encrypted in transit & at rest | HTTPS + AES-256 encryption |
| **User data** | Deletable by user anytime | GDPR compliance, right to be forgotten |
| **API keys** | Secured, never exposed | Environment variables, .env files |
| **Authentication** | Prevent unauthorized access | JWT tokens, refresh tokens |

### Privacy Requirements
| Aspect | Requirement | Implementation |
|--------|-------------|-----------------|
| **Consent** | User explicitly opts in | Checkbox on first login |
| **Transparency** | Explain what's collected | Privacy policy + in-app tooltips |
| **Data minimization** | Collect only what's needed | Emotion labels only, not raw media |
| **Data retention** | Automatic deletion policy | Delete sessions >90 days old (configurable) |
| **No profiling** | Emotion data not used to manipulate | No A/B testing on suppressed emotion |

### Scalability Requirements
| Component | Target Capacity | Scaling Strategy |
|-----------|-----------------|------------------|
| **Concurrent users** | 100+ simultaneous WebRTC sessions | Horizontal scaling with load balancer |
| **Database** | 1M+ session records | PostgreSQL replication + read replicas |
| **File storage** | 1TB+ emotion logs | Cloud storage (S3/GCS) with archival |
| **API throughput** | 10K requests/second | FastAPI async + Kubernetes autoscaling |

---

# 3. TECHNOLOGY STACK (Complete)

## 3.1 Frontend Stack

### Core Framework
```
React 18.2+
├─ TypeScript (type safety)
├─ Vite (fast build tool, replaces CRA)
└─ Node 18 LTS
```

### Real-Time Communication
```
WebRTC (getUserMedia API)
├─ Video capture: navigator.mediaDevices.getUserMedia({video, audio})
├─ Peer connection: RTCPeerConnection (optional for P2P, else WebSocket)
└─ Audio processing: Web Audio API for feature extraction

Socket.io or native WebSocket
├─ Bidirectional communication
├─ Real-time event streaming
└─ Fallback to HTTP long-polling
```

### UI Libraries
```
TailwindCSS 3.x
├─ Utility-first CSS framework
├─ Responsive design out-of-box
└─ Dark mode support

shadcn/ui
├─ Accessible component library built on Radix UI
├─ Pre-built: Button, Input, Card, Dialog, Slider
└─ Customizable with Tailwind

Recharts or Chart.js
├─ Emotion timeline charts
├─ Real-time data visualization
└─ Dashboard analytics
```

### State Management
```
Zustand or Redux Toolkit
├─ WebRTC connection state
├─ Current emotion/stress data
├─ User session info
├─ Message history
└─ UI state (modal open, sidebar visible)
```

### Video/Audio Utilities
```
simple-peer (WebRTC abstraction)
├─ Simplifies WebRTC setup
├─ Handle SDP negotiation
└─ Connection state management

MediaDevices API (native)
├─ getUserMedia for video/audio
├─ Constraints: resolution, framerate
└─ Permissions handling
```

### Development Tools
```
Vite (build tool)
├─ Hot Module Replacement (HMR) for instant feedback
├─ Faster builds than Webpack
└─ Optimized production bundle

ESLint + Prettier
├─ Code linting + formatting
├─ Enforce code style
└─ Pre-commit hooks (husky)

Vitest + React Testing Library
├─ Unit tests
├─ Component tests
└─ Integration tests
```

---

## 3.2 Backend Stack

### Web Framework
```
FastAPI 0.95+
├─ Async Python (asyncio)
├─ Built-in OpenAPI docs (/docs)
├─ Type hints with Pydantic
├─ 10K+ RPS capability
└─ CORS, WebSocket support

Uvicorn
├─ ASGI server (Async Server Gateway Interface)
├─ Single-threaded, high concurrency
├─ Production-grade
```

### Emotion Detection Services

#### A. Face Emotion
```
MediaPipe 0.10+
├─ Face mesh detection: 468 landmarks
├─ Real-time 30 FPS on consumer GPU
├─ Handles: Rotation, occlusion, lighting
├─ No model training needed
└─ Lightweight (80MB)

DeepFace 0.0.71+
├─ Ensemble of 4 pre-trained CNNs
├─ 7-emotion classification
├─ 95%+ accuracy on FER2013
├─ Pre-trained models (no training needed)
└─ Models: VGGFace, Facenet, OpenFace, DeepID

OpenCV 4.8+
├─ Image processing
├─ Frame capture from video stream
├─ Face rectangle drawing
└─ Utility functions
```

#### B. Voice Emotion
```
Librosa 0.10+
├─ Audio feature extraction
├─ MFCC (40 coefficients)
├─ Pitch, energy, ZCR
└─ Noise reduction

SoundFile
├─ Read/write audio files
├─ WAV, FLAC support
└─ 16-bit mono recommended

NumPy + SciPy
├─ Audio signal processing
├─ FFT for frequency analysis
└─ Statistical features

PyTorch 2.0+
├─ Pre-trained audio models
├─ CNN-LSTM architecture
├─ GPU acceleration
└─ Model inference

HuBERT / WavLM (optional)
├─ Pre-trained audio embeddings
├─ Transfer learning for voice emotion
└─ Fine-tuning on IEMOCAP dataset
```

#### C. Text Emotion
```
HuggingFace Transformers 4.35+
├─ BERT, RoBERTa, DistilBERT
├─ Pre-trained on 3.3B words
├─ 768-dimensional embeddings
└─ Pipeline API for easy inference

VADER Sentiment
├─ Rule-based sentiment analysis
├─ Fast (no neural network)
├─ Good for informal text
└─ Handles emoticons, slang

spaCy 3.7+
├─ Named entity recognition (NER)
├─ Extract: Person, Organization, Event, Product
├─ Dependency parsing
└─ Custom trained models (optional)

NLTK 3.8+
├─ Tokenization
├─ Lemmatization
├─ Stopword removal
└─ Corpus data
```

### Fusion & LLM Integration

```
NumPy + Pandas
├─ Attention mechanism implementation
├─ Weighted voting
├─ Data manipulation
└─ Statistical analysis

OpenAI Python SDK
├─ GPT-4, GPT-3.5-turbo API calls
├─ Streaming responses
├─ Token counting
└─ Error handling

Pydantic v2
├─ Data validation
├─ Type hints for API requests/responses
├─ JSON schema generation
└─ Settings management (BaseSettings)
```

### Database
```
PostgreSQL 15+
├─ Relational data (users, sessions, emotions)
├─ ACID compliance
├─ Full-text search
├─ JSONB for flexible schema
└─ Time-series data (emotion logs)

SQLAlchemy 2.0+
├─ ORM (Object Relational Mapping)
├─ Database-agnostic queries
├─ Connection pooling (asyncpg)
└─ Migrations (Alembic)

Alembic
├─ Database schema versioning
├─ Automatic migration scripts
└─ Rollback capability

Redis (optional)
├─ Session cache
├─ Real-time data (current mood)
├─ Rate limiting
└─ Job queue (Celery)
```

### Supporting Services

```
Celery + Redis
├─ Async task queue
├─ Offload heavy computation
├─ Email notifications
└─ Periodic cleanup jobs

APScheduler
├─ Scheduled jobs
├─ Cleanup old sessions
├─ Send daily reports
└─ Burnout risk re-calculation

python-dotenv
├─ Environment variables
├─ API keys (OpenAI, etc.)
├─ Database URL
└─ Debug mode toggle
```

### Development Tools
```
Pytest
├─ Unit tests
├─ Integration tests
├─ Fixtures + mocking
└─ Code coverage reports

Black + isort
├─ Python code formatting
├─ Import organization
└─ Enforce PEP 8 style

Pylint + mypy
├─ Static code analysis
├─ Type checking
├─ Detect bugs early
└─ Pre-commit hooks
```

---

## 3.3 DevOps & Deployment Stack

### Containerization
```
Docker
├─ Dockerfile: Backend + ML services
├─ docker-compose.yml: Local development
├─ Multi-stage builds (reduce image size)
└─ .dockerignore (exclude unnecessary files)

Docker Compose
├─ Run PostgreSQL + FastAPI + Redis locally
├─ Environment variable injection
├─ Network bridging
└─ Volume mounting for development
```

### Cloud Deployment Options

#### Option A: AWS (Recommended)
```
EC2 Instances
├─ t3.large (backend) or t3.xlarge (GPU inference)
├─ Auto Scaling Group for scaling
├─ Security groups + VPC
└─ Estimated: $50-200/month

RDS (PostgreSQL)
├─ Managed database
├─ Automated backups
├─ Read replicas for scaling
└─ Estimated: $20-50/month

S3
├─ Store session logs, dashboards
├─ Media files (optional)
└─ Estimated: <$5/month

ALB (Application Load Balancer)
├─ Route traffic to EC2 instances
├─ SSL/TLS termination
├─ Sticky sessions for WebSocket
└─ Estimated: $15-20/month

CloudFront
├─ CDN for static assets
├─ Cache React bundle, charts
└─ Estimated: <$5/month

TOTAL AWS COST: ~$100-300/month
```

#### Option B: GCP (Alternative)
```
Cloud Run (serverless)
├─ Auto-scales to zero
├─ Pay per request
├─ FastAPI compatible
└─ Estimated: $10-30/month

Cloud SQL (PostgreSQL)
├─ Similar to AWS RDS
├─ Estimated: $15-40/month

Cloud Storage + CDN
├─ Similar to S3 + CloudFront
└─ Estimated: <$5/month

TOTAL GCP COST: ~$30-75/month (cheaper than AWS)
```

#### Option C: Heroku (Simplest for MCA)
```
Heroku Dyno
├─ Standard 1X or 2X
├─ Auto-restart on crash
├─ Easy GitHub integration
└─ Estimated: $7-50/month

Heroku Postgres
├─ Managed database
├─ Estimated: $9-50/month

TOTAL HEROKU COST: ~$20-100/month (easiest, not cheapest)
```

### CI/CD Pipeline
```
GitHub Actions (free for public repos)
├─ Trigger: On push to main/dev
├─ Test: Run pytest suite
├─ Build: Docker image
├─ Deploy: To AWS/GCP/Heroku
└─ Workflows: .github/workflows/*.yml

GitLab CI (if using GitLab)
├─ Similar pipeline
├─ More storage for artifacts
└─ Self-hosted runners available
```

### Monitoring & Logging
```
Sentry
├─ Error tracking
├─ Real-time alerts
├─ Source map support
└─ Free tier: 5,000 errors/month

Datadog (optional)
├─ APM (Application Performance Monitoring)
├─ Real-time metrics
├─ Custom dashboards
└─ Estimated: $50+/month

CloudWatch (AWS) or Stackdriver (GCP)
├─ Built-in logging
├─ Metrics dashboards
└─ Included with cloud provider

Prometheus + Grafana (self-hosted)
├─ Open source alternative
├─ Real-time metrics
└─ Cost: Server hosting (~$20/month)
```

---

## 3.4 Analytics & Dashboard Stack

### Dashboard Tools
```
Streamlit (Quickest for MVP)
├─ Python-based web app
├─ No JavaScript needed
├─ Charts built-in
├─ Rerun on code change
├─ Deployment: streamlit.io (free)

React Dashboard (More control)
├─ Custom React components
├─ Same frontend as chatbot
├─ Recharts for visualizations
├─ More effort but flexible
```

### Data Visualization
```
Recharts
├─ React library
├─ Line charts (emotion over time)
├─ Pie charts (emotion distribution)
├─ Responsive design
└─ TypeScript support

Plotly (Python)
├─ Interactive charts
├─ 3D visualizations
├─ Export to HTML
└─ Works with Streamlit
```

---

# 4. FILE STRUCTURE & ARCHITECTURE

## 4.1 Project Root Structure

```
RIVION/
├── frontend/                          # React application
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/               # UI components
│   │   │   ├── VideoChat/            # Main video interface
│   │   │   │   ├── VideoChat.tsx
│   │   │   │   ├── VideoChat.module.css
│   │   │   │   ├── EmotionDisplay.tsx
│   │   │   │   ├── ChatBox.tsx
│   │   │   │   └── AvatarDisplay.tsx
│   │   │   ├── Dashboard/
│   │   │   │   ├── Dashboard.tsx
│   │   │   │   ├── StressGauge.tsx
│   │   │   │   ├── EmotionTimeline.tsx
│   │   │   │   ├── HistoricalTrends.tsx
│   │   │   │   └── EmotionDistribution.tsx
│   │   │   ├── Common/
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   └── Button.tsx
│   │   │   └── Auth/
│   │   │       ├── Login.tsx
│   │   │       ├── Register.tsx
│   │   │       └── ProtectedRoute.tsx
│   │   ├── pages/                    # Page components
│   │   │   ├── HomePage.tsx
│   │   │   ├── ChatPage.tsx
│   │   │   ├── DashboardPage.tsx
│   │   │   └── SettingsPage.tsx
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useWebRTC.ts
│   │   │   ├── useEmotion.ts
│   │   │   ├── useLocalStorage.ts
│   │   │   └── useAuth.ts
│   │   ├── services/                 # API calls
│   │   │   ├── api.ts               # Axios/fetch wrapper
│   │   │   ├── emotionService.ts
│   │   │   ├── chatService.ts
│   │   │   ├── authService.ts
│   │   │   └── dashboardService.ts
│   │   ├── store/                    # State management (Zustand)
│   │   │   ├── emotionStore.ts
│   │   │   ├── chatStore.ts
│   │   │   └── authStore.ts
│   │   ├── types/                    # TypeScript types
│   │   │   ├── emotion.ts
│   │   │   ├── api.ts
│   │   │   ├── user.ts
│   │   │   └── chat.ts
│   │   ├── utils/                    # Utility functions
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   ├── constants.ts
│   │   │   └── logger.ts
│   │   ├── styles/                   # Global styles
│   │   │   ├── globals.css
│   │   │   ├── variables.css
│   │   │   └── animations.css
│   │   ├── App.tsx
│   │   ├── App.module.css
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── .env.example
│   └── .gitignore
│
├── backend/                           # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # Entry point
│   │   ├── config.py                # Settings/config
│   │   ├── dependencies.py          # Dependency injection
│   │   ├── middleware.py            # CORS, logging, etc.
│   │   │
│   │   ├── routers/                 # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # /api/auth/* endpoints
│   │   │   ├── chat.py              # /api/chat/* endpoints
│   │   │   ├── emotion.py           # /api/emotion/* endpoints
│   │   │   ├── dashboard.py         # /api/dashboard/* endpoints
│   │   │   └── health.py            # /api/health endpoint
│   │   │
│   │   ├── models/                  # Database models (SQLAlchemy ORM)
│   │   │   ├── __init__.py
│   │   │   ├── user.py              # User model
│   │   │   ├── session.py           # Chat session model
│   │   │   ├── emotion_log.py       # Emotion record model
│   │   │   ├── message.py           # Chat message model
│   │   │   └── base.py              # Base model class
│   │   │
│   │   ├── schemas/                 # Pydantic schemas (request/response validation)
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── emotion.py
│   │   │   ├── chat.py
│   │   │   ├── auth.py
│   │   │   └── dashboard.py
│   │   │
│   │   ├── services/                # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── face_emotion_service.py      # MediaPipe + DeepFace
│   │   │   ├── voice_emotion_service.py     # Librosa + PyTorch
│   │   │   ├── text_emotion_service.py      # BERT + spaCy
│   │   │   ├── fusion_service.py            # Multimodal fusion
│   │   │   ├── llm_service.py               # OpenAI API wrapper
│   │   │   ├── auth_service.py              # JWT, password hashing
│   │   │   ├── chat_service.py              # Chat logic
│   │   │   ├── emotion_service.py           # Emotion tracking
│   │   │   └── dashboard_service.py         # Analytics
│   │   │
│   │   ├── utils/                   # Utility functions
│   │   │   ├── __init__.py
│   │   │   ├── logger.py            # Logging setup
│   │   │   ├── validators.py        # Input validation
│   │   │   ├── decorators.py        # Custom decorators
│   │   │   ├── helpers.py           # Helper functions
│   │   │   └── constants.py         # Constants
│   │   │
│   │   ├── db/                      # Database
│   │   │   ├── __init__.py
│   │   │   ├── database.py          # Connection setup
│   │   │   ├── session.py           # Session management
│   │   │   └── init_db.py           # Initialize DB
│   │   │
│   │   ├── ml_models/               # Pre-trained models (cache)
│   │   │   ├── __init__.py
│   │   │   ├── face_models/
│   │   │   │   ├── deepface_models.pth (if custom trained)
│   │   │   │   └── mediapipe_face_detection.tflite
│   │   │   ├── voice_models/
│   │   │   │   └── speech_emotion_model.pt (CNN-LSTM)
│   │   │   └── text_models/
│   │   │       └── bert_emotion_model/ (HF model cache)
│   │   │
│   │   └── websocket/               # WebSocket handlers
│   │       ├── __init__.py
│   │       ├── connection_manager.py
│   │       ├── event_handlers.py
│   │       └── middleware.py
│   │
│   ├── migrations/                  # Alembic database migrations
│   │   ├── versions/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── alembic.ini
│   │
│   ├── tests/                       # Test suite
│   │   ├── __init__.py
│   │   ├── conftest.py              # Pytest fixtures
│   │   ├── unit/
│   │   │   ├── test_emotion_service.py
│   │   │   ├── test_fusion_service.py
│   │   │   ├── test_auth_service.py
│   │   │   └── test_llm_service.py
│   │   ├── integration/
│   │   │   ├── test_api_endpoints.py
│   │   │   ├── test_websocket.py
│   │   │   ├── test_auth_flow.py
│   │   │   └── test_chat_flow.py
│   │   └── e2e/
│   │       └── test_full_workflow.py
│   │
│   ├── requirements.txt              # Python dependencies
│   ├── requirements-dev.txt          # Dev dependencies
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── .env.example
│   ├── .gitignore
│   └── pyproject.toml               # Project metadata
│
├── dashboard/                         # Streamlit app (optional)
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── home.py
│   │   ├── emotion_analysis.py
│   │   ├── stress_tracking.py
│   │   └── settings.py
│   ├── components/
│   │   ├── metrics.py
│   │   ├── charts.py
│   │   └── tables.py
│   ├── config.py
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── config.toml
│   └── .gitignore
│
├── docker-compose.yml               # Local development stack
├── docker-compose.prod.yml          # Production stack
├── .github/
│   └── workflows/
│       ├── test.yml                 # Run tests on PR
│       ├── build.yml                # Build Docker image
│       └── deploy.yml               # Deploy to cloud
├── .gitignore
├── README.md                         # Project documentation
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE
└── ARCHITECTURE.md                   # Detailed architecture docs

```

---

## 4.2 Backend Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Video Stream │  │ Audio Stream │  │ Text Input   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                          │
└─────────┼─────────────────┼─────────────────┼──────────────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘ (WebSocket/REST)
                            │
┌─────────────────────────────▼──────────────────────────────────────────┐
│                    FASTAPI BACKEND                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    API ROUTER LAYER                              │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  /api/auth/*      /api/chat/*     /api/emotion/*  /api/dash/*   │ │
│  │  (JWT tokens)     (Messages)      (Stream data)   (Analytics)   │ │
│  └─────────┬───────────────┬────────────────┬────────────────┬───┘ │
│            │               │                │                │     │
│  ┌─────────▼────────────────▼────────────────▼────────────────▼───┐ │
│  │              MIDDLEWARE & DEPENDENCIES                         │ │
│  │  ├─ CORS handler                                              │ │
│  │  ├─ JWT authentication                                        │ │
│  │  ├─ Rate limiting                                             │ │
│  │  ├─ Request/response logging                                  │ │
│  │  └─ Error handling                                            │ │
│  └─────────┬─────────────────────────────────────────────────┬───┘ │
│            │                                                 │     │
│  ┌─────────▼────────────────────────────────────────────────▼───┐ │
│  │                   SERVICE LAYER                              │ │
│  ├────────────────────────────────────────────────────────────────┤ │
│  │                                                                │ │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐ │ │
│  │  │ Face Emotion    │  │ Voice Emotion    │  │ Text Emotion │ │ │
│  │  │ Service         │  │ Service          │  │ Service      │ │ │
│  │  │                 │  │                  │  │              │ │ │
│  │  │ • MediaPipe     │  │ • Librosa MFCC   │  │ • BERT       │ │ │
│  │  │ • DeepFace      │  │ • PyTorch LSTM   │  │ • spaCy NER  │ │ │
│  │  │ • OpenCV        │  │ • WavLM/HuBERT   │  │ • VADER      │ │ │
│  │  └────────┬────────┘  └────────┬─────────┘  └──────┬───────┘ │ │
│  │           │                    │                   │         │ │
│  │  ┌────────▼────────────────────▼───────────────────▼───────┐ │ │
│  │  │         FUSION ENGINE                                   │ │ │
│  │  │  • Attention-based weighted voting                      │ │ │
│  │  │  • Hidden stress detection                              │ │ │
│  │  │  • Confidence scoring                                   │ │ │
│  │  │  • Temporal smoothing (10-frame MA)                     │ │ │
│  │  └────────┬──────────────────────────────────────────────┘ │ │
│  │           │                                                 │ │
│  │  ┌────────▼──────────────────────────────────────────────┐ │ │
│  │  │         LLM SERVICE                                   │ │ │
│  │  │  • OpenAI API wrapper                                 │ │ │
│  │  │  • Dynamic system prompt (emotion-based)              │ │ │
│  │  │  • Safety checks (crisis detection)                   │ │ │
│  │  │  • Response streaming                                 │ │ │
│  │  └────────┬──────────────────────────────────────────────┘ │ │
│  │           │                                                 │ │
│  │  ┌────────▼──────────────────────────────────────────────┐ │ │
│  │  │    DATABASE & CACHE SERVICES                          │ │ │
│  │  │  • PostgreSQL (persistent data)                       │ │ │
│  │  │  • Redis (session cache, rate limit)                  │ │ │
│  │  │  • SQLAlchemy ORM                                     │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
          │
          │ (REST API, SQL queries, Redis commands)
          │
    ┌─────▼──────────────────────────────────────┐
    │        EXTERNAL SERVICES & DATA             │
    ├──────────────────────────────────────────────┤
    │  • PostgreSQL (primary database)             │
    │  • Redis (cache layer)                       │
    │  • OpenAI API (LLM responses)                │
    │  • Sentry (error tracking)                   │
    │  • AWS S3 (session backups)                  │
    │  • Mailgun (email notifications)             │
    └──────────────────────────────────────────────┘
```

---

# 5. DATABASE DESIGN

## 5.1 Entity Relationship Diagram (ERD)

```
┌─────────────────────────────────┐
│          USER                   │
├─────────────────────────────────┤
│ id (PK, UUID)                   │
│ email (UNIQUE)                  │
│ username                        │
│ password_hash                   │
│ created_at                      │
│ updated_at                      │
│ deleted_at (soft delete)        │
│ privacy_consent (BOOLEAN)       │
│ data_retention_days (INT)       │
└────────────────┬────────────────┘
                 │ (1:N)
                 │ owns
                 │
         ┌───────▼──────────────────────┐
         │      SESSION                 │
         ├───────────────────────────────┤
         │ id (PK, UUID)                │
         │ user_id (FK)                 │
         │ session_start (TIMESTAMP)    │
         │ session_end (TIMESTAMP)      │
         │ total_stress_score (FLOAT)   │
         │ dominant_emotion (VARCHAR)   │
         │ burnout_risk_level (ENUM)    │
         │ notes (TEXT)                 │
         └───────┬──────────┬──────────────┘
                 │          │
            (1:N)│ contains │ (1:N)
                 │          │
    ┌────────────▼───┐  ┌──────▼──────────────┐
    │    MESSAGE     │  │   EMOTION_LOG      │
    ├────────────────┤  ├────────────────────┤
    │ id (PK)        │  │ id (PK)            │
    │ session_id(FK) │  │ session_id (FK)    │
    │ sender (ENUM)  │  │ timestamp          │
    │ │(USER/AI)     │  │ face_emotion       │
    │ content (TEXT) │  │ face_confidence    │
    │ emotion_label  │  │ voice_emotion      │
    │ created_at     │  │ voice_confidence   │
    │ tokens_used    │  │ text_emotion       │
    │ response_time  │  │ text_confidence    │
    └────────────────┘  │ final_emotion      │
                        │ stress_score (0-100)
                        │ hidden_stress      │
                        │ face_features(JSONB)
                        │ voice_features(JSONB)
                        │ text_keywords(JSONB)
                        └────────────────────┘

ADDITIONAL TABLES:
┌────────────────────────┐
│   EMOTION_STATISTICS   │  (aggregated for performance)
├────────────────────────┤
│ id (PK)                │
│ user_id (FK)           │
│ date (DATE)            │
│ avg_stress_score       │
│ emotion_counts (JSONB) │ {"happy": 5, "sad": 2, ...}
│ burnout_risk           │
└────────────────────────┘

┌────────────────────────┐
│   FEEDBACK             │  (user feedback on suggestions)
├────────────────────────┤
│ id (PK)                │
│ session_id (FK)        │
│ suggestion             │ (VARCHAR, e.g., "breathing_exercise")
│ was_helpful (BOOLEAN)  │
│ rating (INT, 1-5)      │
└────────────────────────┘

┌────────────────────────┐
│   AUDIT_LOG            │  (track all API calls)
├────────────────────────┤
│ id (PK)                │
│ user_id (FK)           │
│ action (VARCHAR)       │
│ resource (VARCHAR)     │
│ timestamp              │
│ ip_address             │
│ user_agent             │
└────────────────────────┘
```

## 5.2 Database Schema (SQL)

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP,
    privacy_consent BOOLEAN DEFAULT FALSE,
    data_retention_days INT DEFAULT 90,
    is_active BOOLEAN DEFAULT TRUE
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    total_stress_score FLOAT,
    dominant_emotion VARCHAR(50),
    burnout_risk_level VARCHAR(20), -- LOW, MEDIUM, HIGH
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sender VARCHAR(10) NOT NULL, -- USER or AI
    content TEXT NOT NULL,
    emotion_label VARCHAR(50),
    tokens_used INT,
    response_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Emotion logs table (high-frequency data)
CREATE TABLE emotion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Face emotion
    face_emotion VARCHAR(50),
    face_confidence FLOAT,
    face_arousal FLOAT, -- 0-1
    face_valence FLOAT, -- 0-1
    face_features JSONB, -- {landmarks, action_units, ...}
    
    -- Voice emotion
    voice_emotion VARCHAR(50),
    voice_confidence FLOAT,
    voice_stress FLOAT, -- 0-1
    voice_tone VARCHAR(50), -- calm, stressed, angry, etc.
    voice_features JSONB, -- {pitch, energy, mfcc, ...}
    
    -- Text emotion
    text_emotion VARCHAR(50),
    text_confidence FLOAT,
    text_sentiment FLOAT, -- -1 to 1
    text_keywords JSONB, -- ["exhausted", "overwhelmed"]
    text_risk_level VARCHAR(20), -- NORMAL, MILD, MODERATE, SEVERE
    
    -- Fusion results
    final_emotion VARCHAR(50),
    final_confidence FLOAT,
    stress_score INT, -- 0-100
    hidden_stress_detected BOOLEAN,
    
    INDEX idx_session_time (session_id, timestamp DESC)
);

-- Emotion statistics (aggregated daily)
CREATE TABLE emotion_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    avg_stress_score FLOAT,
    max_stress_score INT,
    emotion_distribution JSONB, -- {"happy": 30, "sad": 20, ...}
    burnout_risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    suggestion_type VARCHAR(100), -- breathing_exercise, task_breakdown, etc.
    was_helpful BOOLEAN,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    user_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_sessions_user_date ON sessions(user_id, session_start DESC);
CREATE INDEX idx_emotion_logs_session ON emotion_logs(session_id);
CREATE INDEX idx_emotion_stats_user_date ON emotion_statistics(user_id, date DESC);

-- Enable full-text search on messages
CREATE INDEX idx_messages_content_fts ON messages USING GIN(to_tsvector('english', content));
```

---

# 6. API ENDPOINTS SPECIFICATION

## 6.1 Authentication Endpoints

```
POST /api/auth/register
├─ Request: { email, username, password, privacy_consent }
├─ Response: { user_id, token, refresh_token }
├─ Status: 201 Created
└─ Errors: 400 (validation), 409 (user exists)

POST /api/auth/login
├─ Request: { email, password }
├─ Response: { user_id, token, refresh_token, expires_in }
├─ Status: 200 OK
└─ Errors: 401 (invalid credentials), 404 (user not found)

POST /api/auth/logout
├─ Request: { token }
├─ Response: { message: "Logged out" }
├─ Status: 200 OK
└─ Auth: Required

POST /api/auth/refresh
├─ Request: { refresh_token }
├─ Response: { token, refresh_token }
├─ Status: 200 OK
└─ Errors: 401 (invalid token)

GET /api/auth/me
├─ Response: { user_id, email, username, created_at }
├─ Status: 200 OK
├─ Auth: Required
└─ Errors: 401 (not authenticated)

PUT /api/auth/profile
├─ Request: { first_name, last_name, privacy_consent }
├─ Response: { user_id, ...updated_fields }
├─ Status: 200 OK
├─ Auth: Required
└─ Errors: 400 (validation)

DELETE /api/auth/account
├─ Request: { password_confirmation }
├─ Response: { message: "Account deleted" }
├─ Status: 200 OK
├─ Auth: Required
└─ Notes: GDPR right to be forgotten (cascade delete all user data)
```

## 6.2 Chat Endpoints

```
POST /api/chat/session/start
├─ Request: { topic: "optional" }
├─ Response: { session_id, timestamp }
├─ Status: 201 Created
├─ Auth: Required
└─ WebSocket: Opens WebSocket connection for real-time chat

WS /api/chat/stream
├─ Connect: Opens bidirectional WebSocket
├─ Listen events:
│  ├─ video_frame: { frame_base64, timestamp }
│  ├─ audio_chunk: { audio_base64, duration_ms }
│  └─ text_message: { content, timestamp }
├─ Emit events:
│  ├─ emotion_update: { emotion, stress_score, confidence }
│  ├─ ai_response: { message, suggestion }
│  └─ error: { code, message }
├─ Auth: JWT token in query params
└─ Heartbeat: Every 30 seconds

POST /api/chat/message
├─ Request: { session_id, content }
├─ Response: { message_id, ai_response, emotion_detected }
├─ Status: 201 Created
├─ Auth: Required
└─ Notes: Alternative to WebSocket (slower, for backup)

POST /api/chat/session/end
├─ Request: { session_id }
├─ Response: { session_summary, stress_trend, recommendations }
├─ Status: 200 OK
├─ Auth: Required
└─ Notes: Close session, compute final stats

GET /api/chat/sessions
├─ Query: { limit: 20, offset: 0, sort: "date" }
├─ Response: { sessions: [...], total: 100 }
├─ Status: 200 OK
├─ Auth: Required
└─ Pagination: offset-limit based

GET /api/chat/session/{session_id}
├─ Response: { session_id, messages: [...], emotions: [...], stats: {...} }
├─ Status: 200 OK
├─ Auth: Required
└─ Errors: 404 (session not found)
```

## 6.3 Emotion Analysis Endpoints

```
POST /api/emotion/analyze-face
├─ Request: { frame_base64 }
├─ Response: { emotion, confidence, arousal, landmarks: [...] }
├─ Status: 200 OK
├─ Auth: Optional
└─ Processing time: <100ms

POST /api/emotion/analyze-voice
├─ Request: { audio_base64, duration_ms }
├─ Response: { emotion, confidence, stress_level, tone }
├─ Status: 200 OK
├─ Auth: Optional
└─ Processing time: <500ms

POST /api/emotion/analyze-text
├─ Request: { text }
├─ Response: { emotion, confidence, sentiment, keywords: [...], risk_level }
├─ Status: 200 OK
├─ Auth: Optional
└─ Processing time: <50ms

POST /api/emotion/fuse
├─ Request: { face_emotion, voice_emotion, text_emotion, confidences }
├─ Response: { final_emotion, stress_score, hidden_stress_detected }
├─ Status: 200 OK
├─ Auth: Optional
└─ Algorithm: Attention-based weighted voting

GET /api/emotion/history/{session_id}
├─ Response: { emotions: [...], timeline: {...}, average_stress }
├─ Status: 200 OK
├─ Auth: Required
└─ Notes: Time-series data for graphs
```

## 6.4 Dashboard Endpoints

```
GET /api/dashboard/summary
├─ Query: { date_range: "7d" | "30d" | "90d" }
├─ Response: {
│    current_stress: 45,
│    burnout_risk: "low",
│    avg_stress_trend: [...],
│    emotion_distribution: {...},
│    recommendations: [...]
│  }
├─ Status: 200 OK
├─ Auth: Required
└─ Performance: Cached, updated hourly

GET /api/dashboard/stress-trend
├─ Query: { days: 30 }
├─ Response: { 
│    data: [
│      { date: "2025-12-01", stress: 45, emotion: "calm" },
│      { date: "2025-12-02", stress: 62, emotion: "stressed" }
│    ]
│  }
├─ Status: 200 OK
├─ Auth: Required
└─ Aggregated per session

GET /api/dashboard/emotion-distribution
├─ Query: { days: 7 }
├─ Response: {
│    happy: 10,
│    sad: 5,
│    angry: 3,
│    neutral: 20,
│    ...
│  }
├─ Status: 200 OK
├─ Auth: Required
└─ Percentage or count

GET /api/dashboard/burnout-risk
├─ Response: {
│    risk_level: "MEDIUM",
│    score: 62,
│    factors: ["increasing_stress", "declining_mood"],
│    recommendations: ["seek_professional_help", "reduce_workload"]
│  }
├─ Status: 200 OK
├─ Auth: Required
└─ Algorithm: Multi-factor analysis

POST /api/dashboard/export
├─ Query: { format: "csv" | "pdf", date_range: "90d" }
├─ Response: Binary file (download)
├─ Status: 200 OK
├─ Auth: Required
└─ Privacy: User data only, anonymized
```

## 6.5 Settings & Preferences Endpoints

```
GET /api/settings/preferences
├─ Response: { 
│    theme: "dark",
│    notification_enabled: true,
│    data_retention_days: 90,
│    privacy_level: "high"
│  }
├─ Status: 200 OK
├─ Auth: Required
└─ User-specific settings

PUT /api/settings/preferences
├─ Request: { theme, notification_enabled, data_retention_days }
├─ Response: { updated_at, preferences }
├─ Status: 200 OK
├─ Auth: Required
└─ Validates each field

DELETE /api/settings/data
├─ Request: { confirm: true }
├─ Response: { message: "All data deleted" }
├─ Status: 200 OK
├─ Auth: Required
├─ GDPR: Right to be forgotten
└─ Irreversible: Cannot undo
```

---

# 7. PHASE-BY-PHASE IMPLEMENTATION

## PHASE 1: MVP (Weeks 1-4) - Face + Text Only

### 7.1.1 Week 1: Setup & Foundation

**Backend Tasks:**
- [ ] Create FastAPI project structure
- [ ] Set up PostgreSQL locally (docker-compose)
- [ ] Implement user registration/login (JWT auth)
- [ ] Create database models (User, Session, Message, EmotionLog)
- [ ] Implement basic CORS + error handling middleware

**Frontend Tasks:**
- [ ] Create React + TypeScript project (Vite)
- [ ] Set up TailwindCSS + shadcn/ui
- [ ] Create login/register pages
- [ ] Set up Zustand store for auth state
- [ ] Implement basic routing (home, chat, settings)

**DevOps:**
- [ ] Initialize Docker Compose setup
- [ ] Configure .env files
- [ ] Set up GitHub repo + .gitignore
- [ ] Create basic CI/CD workflow (GitHub Actions)

**Deliverables:**
- Scaffolded project structure
- Users can register/login
- Auth token management working
- Docker environment ready

---

### 7.1.2 Week 2: Face Emotion Detection

**Backend Tasks:**
- [ ] Create `face_emotion_service.py` (MediaPipe + DeepFace)
  ```python
  import mediapipe as mp
  import cv2
  from deepface import DeepFace
  
  class FaceEmotionService:
      def __init__(self):
          self.mp_face = mp.solutions.face_detection
          self.face_detector = self.mp_face.FaceDetection()
      
      def detect_emotion(self, frame):
          # 1. Detect face in frame
          # 2. Extract face region
          # 3. Use DeepFace to classify emotion
          # 4. Extract landmarks from MediaPipe
          # 5. Calculate arousal (eye opening, jaw tension)
          # 6. Return {emotion, confidence, arousal, landmarks}
          pass
  ```
- [ ] Create `/api/emotion/analyze-face` endpoint
- [ ] Add endpoint tests

**Frontend Tasks:**
- [ ] Implement video stream component (getUserMedia API)
  ```javascript
  const videoRef = useRef<HTMLVideoElement>(null);
  
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    })
    .then(stream => videoRef.current.srcObject = stream);
  }, []);
  ```
- [ ] Create EmotionDisplay component (shows detected emotion + confidence)
- [ ] Implement WebSocket connection (socket.io)
- [ ] Create chat UI (textarea + message list)

**Testing:**
- [ ] Unit tests for face emotion detection
- [ ] Integration test: upload frame → get emotion
- [ ] API tests with Pytest

**Deliverables:**
- Face emotion detection working in real-time
- Video stream displaying in UI
- Emotion labels showing with confidence scores
- Tests passing

---

### 7.1.3 Week 3: Text Emotion & Chat Integration

**Backend Tasks:**
- [ ] Create `text_emotion_service.py` (BERT + spaCy)
  ```python
  from transformers import pipeline
  import spacy
  
  class TextEmotionService:
      def __init__(self):
          self.emotion_classifier = pipeline("text-classification", 
                                             model="j-hartmann/emotion-english-distilroberta-base")
          self.nlp = spacy.load("en_core_web_sm")
      
      def analyze_text(self, text):
          # 1. Classify emotion using BERT
          # 2. Extract entities using spaCy
          # 3. Detect risk keywords ("give up", "hopeless", etc.)
          # 4. Calculate sentiment
          # 5. Return {emotion, confidence, keywords, sentiment, risk_level}
          pass
  ```
- [ ] Create `/api/emotion/analyze-text` endpoint
- [ ] Create `llm_service.py` (OpenAI wrapper)
  ```python
  from openai import AsyncOpenAI
  
  class LLMService:
      def __init__(self, api_key):
          self.client = AsyncOpenAI(api_key=api_key)
      
      async def generate_response(self, user_message, emotion_data):
          # 1. Craft dynamic system prompt based on emotion
          # 2. Include emotion context in user message
          # 3. Call OpenAI API (streaming)
          # 4. Return response stream
          pass
  ```
- [ ] Create `/api/chat/message` endpoint (send message → get response)
- [ ] Implement WebSocket event handlers (text messages)
- [ ] Create `chat_service.py` (message persistence, history)

**Frontend Tasks:**
- [ ] Implement ChatBox component (textarea + message list)
- [ ] Create Message component (display user/AI messages)
- [ ] Implement WebSocket text message sending/receiving
- [ ] Add emotion tags to messages (show detected emotion)
- [ ] Create conversation history sidebar

**Testing:**
- [ ] Test BERT emotion classification
- [ ] Test LLM integration with mock responses
- [ ] Test chat flow (send message → get response)
- [ ] Test emotion persistence to database

**Deliverables:**
- Text-based emotion detection working
- Chat messages being sent/received
- AI responses generated with OpenAI API
- Chat history persistent in database

---

### 7.1.4 Week 4: Fusion + Basic Stress Scoring

**Backend Tasks:**
- [ ] Create `fusion_service.py` (combine face + text emotions)
  ```python
  import numpy as np
  
  class FusionService:
      def fuse_emotions(self, face_emotion, text_emotion, face_conf, text_conf):
          # Attention-based weighted voting
          # Calculate stress score (0-100)
          # Detect contradictions (hidden stress)
          pass
      
      def calculate_stress_score(self, face_emotion, text_emotion, 
                                 face_conf, text_conf, voice_arousal=None):
          # Stress = arousal (from face) + negativity (from text)
          # Score: 0-100
          pass
  ```
- [ ] Create `/api/emotion/fuse` endpoint
- [ ] Implement stress score calculation + storage
- [ ] Create dashboard backend (aggregated stats)

**Frontend Tasks:**
- [ ] Create EmotionDisplay component (shows final fused emotion)
- [ ] Implement stress gauge (circular, 0-100)
- [ ] Create basic Dashboard page (stress history, last 7 sessions)
- [ ] Add real-time emotion updates to WebSocket

**Testing:**
- [ ] Test fusion logic (different combinations)
- [ ] Test stress score edge cases
- [ ] Test dashboard data aggregation
- [ ] End-to-end: video + text → fused emotion → stress score

**Deliverables:**
- Face + text fusion working
- Stress score displayed in real-time
- Dashboard showing historical data
- MVP complete: can have a conversation, see emotions detected

---

## PHASE 2: Voice Emotion (Weeks 5-6)

### 7.2.1 Week 5: Voice Emotion Detection

**Backend Tasks:**
- [ ] Create `voice_emotion_service.py` (Librosa + PyTorch)
  ```python
  import librosa
  import numpy as np
  import torch
  
  class VoiceEmotionService:
      def __init__(self, model_path):
          self.model = self.load_model(model_path)
      
      def extract_features(self, audio_data, sr=16000):
          # MFCC, pitch, energy, ZCR
          mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
          pitch = librosa.yin(audio_data, fmin=80, fmax=400, sr=sr)
          return {mfcc, pitch, energy, zcr}
      
      def classify_emotion(self, audio_data):
          features = self.extract_features(audio_data)
          emotion = self.model.predict(features)
          return emotion
  ```
- [ ] Set up pre-trained voice emotion model (if available)
  - Option 1: Use HuBERT + fine-tune on IEMOCAP
  - Option 2: Pre-trained SER model from GitHub
- [ ] Create `/api/emotion/analyze-voice` endpoint
- [ ] Implement audio streaming from frontend
- [ ] Create WebSocket audio event handler

**Frontend Tasks:**
- [ ] Implement audio stream capture (Web Audio API)
  ```javascript
  const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(mediaStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  
  processor.onaudioprocess = (event) => {
    const audioData = event.inputBuffer.getChannelData(0);
    // Send to backend every 2-5 seconds
  };
  ```
- [ ] Add audio visualization (waveform display)
- [ ] Implement 2-5 second audio buffering + sending

**Testing:**
- [ ] Test voice emotion on IEMOCAP samples
- [ ] Test audio streaming latency
- [ ] Test end-to-end: speak → detect emotion

**Deliverables:**
- Voice emotion detection functional
- Audio streaming working
- 3-modality fusion (face + voice + text) working

---

### 7.2.2 Week 6: Advanced Fusion + Testing

**Backend Tasks:**
- [ ] Refine fusion service to use attention mechanism (optional)
- [ ] Implement hidden stress detection (contradiction flagging)
- [ ] Optimize inference (batch processing, model caching)
- [ ] Comprehensive testing on IEMOCAP dataset

**Frontend Tasks:**
- [ ] Enhance stress gauge to show modality breakdown
- [ ] Add emotion source indication (face: 80%, voice: 20%, text: ...)
- [ ] Real-time emotion update visualization

**Testing & Validation:**
- [ ] Accuracy testing on IEMOCAP (target: 88%+ for face+voice)
- [ ] Full multimodal accuracy (target: 92%+)
- [ ] Load testing (concurrent users, message rate)
- [ ] Integration tests covering full workflow

**Deliverables:**
- Full 3-modality system working
- Accuracy benchmarks documented
- Performance optimized

---

## PHASE 3: Advanced Features & Production (Weeks 7-12)

### 7.3.1 Week 7-8: Burnout Prediction + Multi-Session Tracking

**Backend Tasks:**
- [ ] Implement multi-session trend analysis
- [ ] Create burnout risk calculation algorithm
  ```python
  def calculate_burnout_risk(sessions_history):
      # Multi-dimensional:
      # 1. Stress trend (increasing = bad)
      # 2. Mood average (negative = bad)
      # 3. Emotion variance (high = bad)
      # 4. Session frequency (too many = bad)
      # Risk: LOW (<40), MEDIUM (40-70), HIGH (>70)
      pass
  ```
- [ ] Implement session aggregation (daily stats)
- [ ] Create alerts for escalating stress/burnout

**Frontend Tasks:**
- [ ] Create burnout risk card (dashboard)
- [ ] Implement trend visualization (line chart, stress over time)
- [ ] Add alerts/notifications for burnout risk

**Deliverables:**
- Burnout risk prediction working
- Multi-session tracking + trends
- Alerts implemented

---

### 7.3.2 Week 9-10: Dashboard + Analytics

**Backend Tasks:**
- [ ] Create `/api/dashboard/*` endpoints (all analytics)
- [ ] Implement data aggregation (daily/weekly summaries)
- [ ] Add export functionality (CSV/PDF)

**Frontend Tasks:**
- [ ] Build full Dashboard page
  - Stress trend chart
  - Emotion distribution pie chart
  - Burnout risk indicator
  - Recent sessions list
  - Recommendations
- [ ] Add filters (date range, emotion type)
- [ ] Implement responsive design

**OR Streamlit Alternative:**
```python
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="RIVION Dashboard", layout="wide")

user_data = fetch_user_data(user_id)
stress_history = user_data['stress_timeline']
emotions = user_data['emotions']

st.metric("Current Stress", stress_history[-1], delta=...)
st.line_chart(stress_history)
st.plotly_chart(px.pie(values=emotions, names=emotions.keys()))
```

**Deliverables:**
- Full analytics dashboard
- Export functionality
- Professional-grade visualizations

---

### 7.3.3 Week 11-12: Deployment + Documentation

**DevOps:**
- [ ] Set up production PostgreSQL (AWS RDS or GCP Cloud SQL)
- [ ] Create production Docker setup
- [ ] Configure CI/CD pipeline (auto-deploy on push)
- [ ] Set up monitoring (Sentry, CloudWatch, Datadog)
- [ ] Configure backups + disaster recovery

**Deployment Options:**
- [ ] **AWS** (EC2 + RDS + ALB)
- [ ] **GCP** (Cloud Run + Cloud SQL)
- [ ] **Heroku** (easiest for demo)

**Documentation:**
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Architecture documentation
- [ ] Deployment guide
- [ ] User guide + privacy policy
- [ ] Troubleshooting guide

**Final Testing:**
- [ ] End-to-end testing (all features)
- [ ] Load testing (100+ concurrent users)
- [ ] Security testing (OWASP top 10)
- [ ] Penetration testing

**Deliverables:**
- Deployed system (live URL)
- Full documentation
- Ready for MCA evaluation

---

# 8. DEVELOPMENT ENVIRONMENT SETUP

## 8.1 Local Development Setup (Complete)

### Prerequisites
```bash
# System requirements
├─ OS: Ubuntu 20.04+, macOS 12+, or Windows 11 (with WSL2)
├─ RAM: 16GB minimum (32GB recommended for ML inference)
├─ GPU: Optional but recommended (NVIDIA RTX 3060 or better)
├─ Disk: 50GB SSD (for models + databases)
└─ Internet: 50+ Mbps

# Install System Dependencies
## Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
                     nodejs npm git postgresql postgresql-contrib \
                     ffmpeg libsndfile1 libportaudio2

## macOS (using Homebrew)
brew install python@3.11 node@18 postgresql ffmpeg portaudio
brew services start postgresql

## Windows (using Chocolatey, in PowerShell as Admin)
choco install python nodejs postgresql ffmpeg vcredist140
```

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/RIVION.git
cd RIVION
```

### Step 2: Backend Setup
```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/dev tools

# Create .env file
cp .env.example .env
# Edit .env with your settings:
# DATABASE_URL=postgresql://user:password@localhost:5432/rivion_db
# OPENAI_API_KEY=sk-...
# JWT_SECRET=your-secret-key

# Initialize database
python -m alembic upgrade head  # Run migrations

# Or if first time setup
python -c "from app.db.init_db import init_db; init_db()"
```

### Step 3: Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env
# Edit .env:
# VITE_API_URL=http://localhost:8000
# VITE_WS_URL=ws://localhost:8000

# Development server
npm run dev  # Runs on http://localhost:5173
```

### Step 4: Docker Setup
```bash
cd ..

# Create docker-compose environment
cp docker-compose.yml docker-compose.override.yml

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db

# Stop services
docker-compose down
```

---

## 8.2 Database Setup

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE rivion_db;
CREATE USER rivion_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE rivion_db TO rivion_user;

# Exit psql
\q

# Run migrations
cd backend
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

# Verify tables
psql -U rivion_user -d rivion_db -c "\dt"
```

---

## 8.3 Model Downloads

```bash
# Create models directory
mkdir -p backend/app/ml_models/{face_models,voice_models,text_models}

# Download DeepFace models (automatic on first use)
python -c "from deepface import DeepFace; DeepFace.build_model('Emotion')"

# Download MediaPipe face detection (automatic)
python -c "import mediapipe as mp; mp.solutions.face_detection"

# Download BERT model (automatic, first inference slower)
python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Voice emotion model (if not using pre-trained)
# Download from: https://github.com/audeering/w2v2-how-to (or similar)
# Place in: backend/app/ml_models/voice_models/
```

---

## 8.4 IDE Setup

### VS Code Configuration (.vscode/settings.json)
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "[typescript]": {
    "editor.formatOnSave": true
  },
  "editor.defaultFormatter": "esbenp.prettier-vscode"
}
```

### VS Code Extensions
```
# Install these extensions:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)
- Thunder Client (rangav.vscode-thunder-client) [API testing]
- SQLTools (mtxr.sqltools)
- PostgreSQL (ckolkman.vscode-postgres)
```

---

# 9. DEPLOYMENT & DEVOPS

## 9.1 Docker Setup (Complete)

### Dockerfile (Backend)
```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Build dependencies in builder stage
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Add to PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile (Frontend)
```dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend .
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Install serve to run SPA
RUN npm install -g serve

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
```

### docker-compose.yml (Development)
```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    container_name: rivion_db
    environment:
      POSTGRES_DB: rivion_db
      POSTGRES_USER: rivion_user
      POSTGRES_PASSWORD: rivion_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rivion_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: rivion_redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    container_name: rivion_backend
    environment:
      DATABASE_URL: postgresql://rivion_user:rivion_password@db:5432/rivion_db
      REDIS_URL: redis://redis:6379
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      JWT_SECRET: ${JWT_SECRET}
      DEBUG: "true"
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app  # Hot reload in development
    command: >
      sh -c "alembic upgrade head &&
             uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

  frontend:
    build: ./frontend
    container_name: rivion_frontend
    ports:
      - "3000:3000"
    environment:
      VITE_API_URL: http://localhost:8000
      VITE_WS_URL: ws://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app  # Hot reload
      - /app/node_modules

volumes:
  postgres_data:
```

---

## 9.2 Deployment to AWS

### Step 1: Prepare
```bash
# Build production images
docker-compose build

# Push to ECR (Amazon Container Registry)
aws ecr get-login-password --region us-east-1 | docker login \
  --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag rivion_backend:latest \
  123456789.dkr.ecr.us-east-1.amazonaws.com/rivion-backend:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/rivion-backend:latest

# Repeat for frontend
```

### Step 2: Infrastructure as Code (Terraform)

```hcl
# main.tf

provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "rivion" {
  cidr_block = "10.0.0.0/16"
  tags = { Name = "rivion-vpc" }
}

# RDS Database
resource "aws_db_instance" "rivion_db" {
  identifier       = "rivion-db"
  engine           = "postgres"
  engine_version   = "15.1"
  instance_class   = "db.t3.micro"
  allocated_storage = 20
  db_name          = "rivion_db"
  username         = "rivion_user"
  password         = random_password.db_password.result
  publicly_accessible = false
  skip_final_snapshot = true  # For development only
}

# EC2 Instance (Backend)
resource "aws_instance" "backend" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"
  
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    db_host = aws_db_instance.rivion_db.endpoint
    db_user = aws_db_instance.rivion_db.master_username
    db_pass = random_password.db_password.result
  }))
  
  tags = { Name = "rivion-backend" }
}

# Application Load Balancer
resource "aws_lb" "rivion" {
  name               = "rivion-alb"
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

# Output public IP
output "backend_url" {
  value = aws_lb.rivion.dns_name
}
```

### Step 3: Deploy
```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply
terraform apply

# Output deployed URL
terraform output backend_url
```

---

## 9.3 CI/CD Pipeline (GitHub Actions)

### .github/workflows/deploy.yml
```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REGISTRY: 123456789.dkr.ecr.us-east-1.amazonaws.com

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r backend/requirements-dev.txt
      
      - name: Run tests
        run: |
          pytest backend/tests -v --cov
      
      - name: Run linting
        run: |
          black --check backend/
          pylint backend/app

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build, tag, and push backend image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/rivion-backend:$IMAGE_TAG backend/
          docker push $ECR_REGISTRY/rivion-backend:$IMAGE_TAG

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to AWS
        run: |
          terraform init
          terraform plan -var="image_tag=${{ github.sha }}"
          terraform apply -auto-approve
```

---

# 10. TESTING STRATEGY

## 10.1 Unit Tests (Backend)

```python
# tests/unit/test_face_emotion_service.py
import pytest
from app.services.face_emotion_service import FaceEmotionService

@pytest.fixture
def face_service():
    return FaceEmotionService()

def test_detect_emotion_happy(face_service):
    # Use pre-recorded happy face frame
    frame = load_test_frame("happy.jpg")
    emotion = face_service.detect_emotion(frame)
    
    assert emotion["emotion"] == "happy"
    assert emotion["confidence"] > 0.7

def test_detect_emotion_no_face(face_service):
    # Blank frame
    frame = np.zeros((640, 480, 3))
    emotion = face_service.detect_emotion(frame)
    
    assert emotion["emotion"] == "unknown"
    assert emotion["confidence"] == 0.0
```

## 10.2 Integration Tests

```python
# tests/integration/test_chat_flow.py
@pytest.mark.asyncio
async def test_full_chat_flow(client, user_token):
    # 1. Start session
    response = client.post("/api/chat/session/start", 
                          headers={"Authorization": f"Bearer {user_token}"})
    session_id = response.json()["session_id"]
    
    # 2. Send message
    response = client.post("/api/chat/message",
                          json={"session_id": session_id, "content": "I'm stressed"},
                          headers={"Authorization": f"Bearer {user_token}"})
    assert response.status_code == 201
    assert "ai_response" in response.json()
    
    # 3. Verify emotion was detected
    response = client.get(f"/api/emotion/history/{session_id}",
                         headers={"Authorization": f"Bearer {user_token}"})
    emotions = response.json()["emotions"]
    assert len(emotions) > 0
    assert emotions[-1]["text_emotion"] is not None
```

## 10.3 E2E Tests (Frontend)

```javascript
// cypress/e2e/chat.spec.js
describe('Chat Flow', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000');
    cy.login('test@example.com', 'password123');
  });

  it('should send message and receive response', () => {
    cy.get('[data-testid=chat-input]').type('I feel anxious');
    cy.get('[data-testid=send-button]').click();
    
    cy.contains('I feel anxious').should('be.visible');
    cy.get('[data-testid=ai-message]').should('be.visible');
  });

  it('should display emotion detection', () => {
    cy.get('[data-testid=emotion-display]').should('contain', 'emotion');
    cy.get('[data-testid=stress-gauge]').should('be.visible');
  });
});
```

---

# 11. BEST PRACTICES & STANDARDS

## 11.1 Code Organization Principles

```
├─ Single Responsibility: Each module does ONE thing
├─ DRY (Don't Repeat Yourself): Reusable functions/components
├─ SOLID Principles: For backend services
├─ Dependency Injection: Loose coupling
├─ Asyncio: Always async I/O operations
└─ Type Hints: Full type annotations (mypy)
```

## 11.2 API Design Standards

```
├─ RESTful endpoints (nouns, not verbs)
├─ Consistent response format:
│  {
│    "status": "success" | "error",
│    "data": {...},
│    "error": {...},
│    "timestamp": "2025-12-05T..."
│  }
├─ Proper HTTP status codes (201, 400, 401, 404, 500)
├─ Pagination (offset + limit) for lists
├─ Rate limiting (429 if exceeded)
└─ API versioning (/api/v1/*, /api/v2/*)
```

## 11.3 Security Practices

```
├─ Never log sensitive data (passwords, API keys, tokens)
├─ Use environment variables for secrets
├─ Validate all user inputs
├─ Use parameterized queries (SQL injection prevention)
├─ CORS: Whitelist specific origins
├─ HTTPS in production (not HTTP)
├─ JWT expiration: 1 hour (refresh token: 7 days)
├─ Password hashing: bcrypt (not plain text)
├─ Rate limiting: 100 requests/minute per IP
└─ OWASP Top 10 compliance
```

## 11.4 Performance Optimization

```
├─ Model caching (load once, reuse)
├─ Database indexing (on frequently queried fields)
├─ Connection pooling (reuse DB connections)
├─ Redis caching (session data, aggregated stats)
├─ Async all I/O (never block with sleep())
├─ Batch processing (group similar requests)
├─ CDN for static assets (React bundle, CSS)
├─ Database query optimization (explain analyze)
└─ Load testing (simulate 100+ concurrent users)
```

---

# 12. TIMELINE & RESOURCE ALLOCATION

## 12.1 Weekly Breakdown (12 Weeks Total)

| Week | Phase | Backend (hrs) | Frontend (hrs) | DevOps (hrs) | Total |
|------|-------|--------------|---|---|---|
| 1 | Setup | 15 | 15 | 10 | 40 |
| 2 | Face Emotion | 20 | 20 | 5 | 45 |
| 3 | Text + Chat | 25 | 20 | 5 | 50 |
| 4 | Fusion | 15 | 15 | 5 | 35 |
| 5 | Voice (Pt1) | 25 | 15 | 5 | 45 |
| 6 | Voice (Pt2) | 20 | 10 | 5 | 35 |
| 7 | Burnout | 20 | 15 | 5 | 40 |
| 8 | Advanced Features | 15 | 20 | 5 | 40 |
| 9 | Dashboard | 10 | 30 | 5 | 45 |
| 10 | Testing | 20 | 15 | 5 | 40 |
| 11 | Deployment | 15 | 10 | 25 | 50 |
| 12 | Documentation | 10 | 10 | 5 | 25 |
| **TOTAL** | | **225 hrs** | **195 hrs** | **85 hrs** | **505 hrs** |

**Average: 20.2 hours/week (part-time MCA project)**

---

## 12.2 Resource Requirements

### Hardware
- Laptop: 16GB RAM, SSD (your own)
- GPU (optional): NVIDIA RTX 3060+ (rent from Paperspace/Lambda Labs: $10-30/month)

### Software (Free/Open Source)
- Python, Node.js, PostgreSQL, Redis (all free)
- FastAPI, React, PyTorch, HuggingFace (all free)

### Paid Services (Recommended Budget)
| Service | Purpose | Cost/Month |
|---------|---------|-----------|
| **OpenAI API** | LLM responses | $20-50 |
| **AWS/GCP** | Cloud hosting | $50-150 |
| **Sentry** | Error tracking | $0 (free tier) |
| **GitHub Pro** | Private repos | $4 |
| **Domain** | Custom domain | $10-15 |
| **SSL Certificate** | HTTPS | $0 (free: Let's Encrypt) |
| **TOTAL** | | ~$100-200/month |

---

# 13. TROUBLESHOOTING GUIDE

## 13.1 Common Backend Issues

### Issue 1: CUDA Out of Memory (GPU)
```python
# Error: torch.cuda.OutOfMemoryError

# Solutions:
1. Reduce batch size in inference
2. Use smaller models (DistilBERT instead of BERT)
3. Offload to CPU temporarily
4. Restart GPU memory with: torch.cuda.empty_cache()
```

### Issue 2: Database Connection Timeouts
```python
# Error: psycopg2.OperationalError: could not connect

# Solutions:
1. Check PostgreSQL is running: `systemctl status postgresql`
2. Verify DATABASE_URL in .env
3. Reset connection pool: `db.dispose()` in SQLAlchemy
4. Increase connection timeout: `pool_pre_ping=True`
```

### Issue 3: WebSocket Connection Fails
```
# Error: WebSocket connection failed

# Solutions:
1. Check CORS headers match frontend origin
2. Verify WebSocket URL in frontend .env
3. Ensure backend listening on correct port
4. Check firewall/network rules allow WebSocket
```

---

## 13.2 Common Frontend Issues

### Issue 1: getUserMedia Permission Denied
```javascript
// Error: NotAllowedError: Permission denied

// Solutions:
1. User must grant camera/mic permission (browser prompt)
2. HTTPS required in production (not localhost:3000)
3. Clear browser permissions: Settings → Privacy → Clear site data
```

### Issue 2: API Calls Return 401 Unauthorized
```javascript
// Solution:
const token = localStorage.getItem('auth_token');
if (!token) {
  // Not logged in
  redirectToLogin();
}

// Include token in headers:
fetch('/api/endpoint', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

---

# APPENDIX: CODE TEMPLATES

## A1. FastAPI Main Entry Point (main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.routers import auth, chat, emotion, dashboard
from app.middleware import ErrorHandlingMiddleware
from app.db.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting RIVION backend...")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="RIVION API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZIPMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ErrorHandlingMiddleware)

# Routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(emotion.router, prefix="/api", tags=["emotion"])
app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## A2. React Chat Component (ChatPage.tsx)

```typescript
import React, { useEffect, useRef, useState } from 'react';
import { useWebRTC } from '@/hooks/useWebRTC';
import { useEmotion } from '@/hooks/useEmotion';
import { VideoChat } from '@/components/VideoChat/VideoChat';
import { ChatBox } from '@/components/VideoChat/ChatBox';
import { EmotionDisplay } from '@/components/VideoChat/EmotionDisplay';

export const ChatPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { video, sendMessage } = useWebRTC();
  const { emotion, stressScore, hidden_stress } = useEmotion();

  return (
    <div className="grid grid-cols-2 gap-6 p-6 h-screen">
      {/* Video Section */}
      <div className="col-span-1">
        <VideoChat videoRef={videoRef} emotion={emotion} />
        <EmotionDisplay 
          emotion={emotion}
          stressScore={stressScore}
          hiddenStress={hidden_stress}
        />
      </div>

      {/* Chat Section */}
      <div className="col-span-1">
        <ChatBox onSendMessage={sendMessage} />
      </div>
    </div>
  );
};
```

---

*This completes the comprehensive RIVION implementation report.*

**Total Documentation: ~15,000 words covering all aspects from requirements to deployment.**

**Next Steps:**
1. Customize for your specific needs
2. Adapt timelines based on your team size
3. Start with Phase 1 MVP (face + text)
4. Iterate and improve based on testing
5. Document your progress for MCA evaluation
