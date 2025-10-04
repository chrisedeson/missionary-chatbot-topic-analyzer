# BYU Pathway Missionary Chatbot Topic Analyzer

A modern dashboard application for analyzing and managing student chatbot questions with hybrid topic discovery and classification.

## 🚀 Features

### Public Dashboard
- Interactive analytics with charts and filters
- Time-based filters (daily, weekly, monthly, custom ranges)
- Geographic filters (country, state) 
- Topic-based filtering and similarity score ranges
- Central interactive table with CSV export
- Recent insights and metrics display

### Developer Tools (Password Protected)
- Questions CSV upload with data cleaning and validation
- Google Sheets synchronization
- Hybrid analysis pipeline (similarity + clustering)
- Database management with confirmation dialogs
- Real-time progress monitoring via Server-Sent Events
- Analysis history and results management

## 🏗️ Architecture

- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS v4, shadcn/ui
- **Backend**: FastAPI with Python, async job processing
- **Database**: PostgreSQL with pgvector extension (Neon)
- **ORM**: Prisma for database operations
- **Queue**: Redis + RQ for background jobs
- **Auth**: Simple password-based developer authentication

## 📋 Prerequisites

- Node.js 18+ and pnpm
- Python 3.11+
- PostgreSQL with pgvector extension
- Redis (for background jobs)
- OpenAI API key
- Google Service Account (for Sheets integration)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd missionary-chatbot-topic-analyzer
```

### 2. Backend Setup

```bash
cd backend

# Install dependencies
make install

# Create environment file from template
make create-env

# Edit .env file with your configuration
nano .env

# Generate Prisma client and setup database
make generate
make migrate

# Start development server
make dev
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Create environment file from template
cp .env.template .env.local

# Edit .env.local with your API URL
nano .env.local

# Start development server
pnpm dev
```

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## ⚙️ Configuration

### Backend Environment Variables

Key configuration variables (see `.env.template` for complete list):

```bash
# Database
DATABASE_URL="postgresql://user:pass@host:5432/db"

# OpenAI (exact settings from hybrid analysis)
OPENAI_API_KEY="your-key"
EMBEDDING_MODEL="text-embedding-3-small"
CHAT_MODEL="gpt-5-nano"
SIMILARITY_THRESHOLD=0.70

# Google Sheets
GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----..."
QUESTIONS_SHEET_ID="1KIu4W9-BYRpZKxrpoWy6qpCBXjSDeRRmKek6q71wTRE"

# Developer Auth
DEV_PASSWORD="pathway2024"
```

### Frontend Environment Variables

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## 🔄 Data Processing Pipeline

### 1. Similarity-based Classification
- Matches new questions to existing topics using embeddings
- Questions above similarity threshold (0.70) → classified to existing topics
- Uses text-embedding-3-small (1536 dimensions)

### 2. Clustering-based Discovery  
- Groups remaining questions using UMAP + HDBSCAN
- Discovers new topic patterns automatically
- Generates topic names using GPT-5-nano

### 3. Database Storage
- Stores embeddings directly in PostgreSQL with pgvector
- Caches embeddings to avoid re-computation
- Tracks analysis runs and results

## 📊 Google Sheets Integration

The application integrates with Google Sheets for data management:

- **Questions Sheet**: Read/write access for uploading new questions
- **Topics Sheet**: Read-only access for existing topic-subtopic-questions

### Data Format

Expected columns for questions:
- `Date`: Question timestamp
- `Country`: Student's country
- `User Language`: Language of the question
- `State`: State/region (for supported countries)
- `Question`: The actual question text

## 🔧 Development

### Backend Commands

```bash
make dev          # Start development server
make test         # Run tests
make lint         # Run linting
make format       # Format code
make clean        # Clean up environment
```

### Frontend Commands

```bash
pnpm dev          # Start development server
pnpm build        # Build for production
pnpm lint         # Run linting
pnpm type-check   # Type checking
```

### Database Operations

```bash
# Generate Prisma client after schema changes
npx prisma generate

# Push schema changes to database
npx prisma db push

# View database in Prisma Studio
npx prisma studio
```

## 🔐 Authentication

Simple password-based authentication for developers:
- Default password: `pathway2024` (configurable via `DEV_PASSWORD`)
- Frontend shows "Sign in as Developer" button
- Backend validates password for protected endpoints

## 📁 Output Files

The analysis pipeline generates three CSV files:

1. **Similar Questions**: Questions matched to existing topics with similarity scores
2. **New Topics**: Newly discovered topics with representative questions
3. **Complete Review**: All questions with topic assignments for review

## 🚨 Error Handling

- Robust data validation and cleaning for malformed CSV files
- Handles "kwargs" error rows from Langfuse logs
- Graceful fallbacks for missing data
- Developer-friendly error messages with detailed logging
- User-friendly error states for the dashboard

## 📈 Monitoring

- Real-time progress monitoring via Server-Sent Events
- Analysis job history and status tracking
- Database metrics and statistics
- OpenAI API usage monitoring

## 🔄 Deployment

### Using Docker

```bash
# Backend
cd backend
docker build -t byu-pathway-backend .
docker run -p 8000:8000 --env-file .env byu-pathway-backend

# Frontend  
cd frontend
docker build -t byu-pathway-frontend .
docker run -p 3000:3000 byu-pathway-frontend
```

### Environment Setup

1. Set up PostgreSQL with pgvector extension
2. Configure Redis for background jobs
3. Set up Google Service Account with Sheets API access
4. Configure OpenAI API access
5. Set environment variables for production

## 🤝 Contributing

1. Follow the existing code style and structure
2. Add tests for new functionality
3. Update documentation for significant changes
4. Use the provided Makefile commands for development

## 📄 License

Apache License 2.0 - see LICENSE file for details.