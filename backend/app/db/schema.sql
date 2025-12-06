-- USERS
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    privacy_consent BOOLEAN DEFAULT FALSE,
    data_retention_days INT DEFAULT 90,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_email ON users (email);


-- SESSIONS
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP WITH TIME ZONE,
    total_stress_score FLOAT,
    dominant_emotion VARCHAR(50),
    burnout_risk_level VARCHAR(20),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_user_date
    ON sessions (user_id, session_start DESC);


-- MESSAGES
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sender VARCHAR(10) NOT NULL, -- 'USER' or 'AI'
    content TEXT NOT NULL,
    emotion_label VARCHAR(50),
    tokens_used INT,
    response_time_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search index on content
CREATE INDEX idx_messages_content_fts
    ON messages USING GIN (to_tsvector('english', content));


-- EMOTION LOGS (high-frequency, per-session)
CREATE TABLE emotion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Face
    face_emotion VARCHAR(50),
    face_confidence FLOAT,
    face_arousal FLOAT,
    face_valence FLOAT,
    face_features JSONB,

    -- Voice
    voice_emotion VARCHAR(50),
    voice_confidence FLOAT,
    voice_stress FLOAT,
    voice_tone VARCHAR(50),
    voice_features JSONB,

    -- Text
    text_emotion VARCHAR(50),
    text_confidence FLOAT,
    text_sentiment FLOAT,
    text_keywords JSONB,
    text_risk_level VARCHAR(20),

    -- Fusion
    final_emotion VARCHAR(50),
    final_confidence FLOAT,
    stress_score INT,
    hidden_stress_detected BOOLEAN
);

CREATE INDEX idx_emotion_logs_session_time
    ON emotion_logs (session_id, timestamp DESC);

CREATE INDEX idx_emotion_logs_session
    ON emotion_logs (session_id);


-- EMOTION STATISTICS (aggregated per user per day)
CREATE TABLE emotion_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    avg_stress_score FLOAT,
    max_stress_score INT,
    emotion_distribution JSONB,
    burnout_risk_level VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, date)
);

CREATE INDEX idx_emotion_stats_user_date
    ON emotion_statistics (user_id, date DESC);


-- FEEDBACK (per session)
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    suggestion_type VARCHAR(100),
    was_helpful BOOLEAN,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    user_comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- AUDIT LOG
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(64),
    user_agent TEXT
);

CREATE INDEX idx_audit_user_time
    ON audit_log (user_id, timestamp DESC);
