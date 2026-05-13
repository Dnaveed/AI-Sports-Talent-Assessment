# AthleteAI - AI-Powered Fitness Assessment Platform

AthleteAI is an AI-powered fitness assessment platform designed to provide objective, scalable, and data-driven evaluation of physical fitness performance. The platform enables authorities (academies, training centers, recruitment bodies) to conduct structured fitness assessments, while athletes can participate, receive automated performance analysis, and track their progress over time.

## Problem Statement

Traditional fitness assessment methods rely heavily on manual observation by trainers or evaluators, leading to:
- **Inconsistency**: Subjective evaluation varies between assessors
- **Limited Scalability**: Manual assessment of large groups is time-consuming
- **Delayed Feedback**: Athletes wait for results and lack detailed performance insights
- **No Historical Tracking**: Difficult to measure improvement over time

## Solution

AthleteAI addresses these challenges by providing:
- **Automated AI Analysis**: Computer vision-based pose estimation analyzes exercise videos
- **Objective Scoring**: Consistent, rule-based evaluation of form and performance
- **Instant Feedback**: Real-time rep counting, form scoring, and quality assessment
- **Progress Tracking**: Historical data, badges, trends, and goal-based progress monitoring
- **Scalable Assessments**: Authorities can schedule tests for multiple participants simultaneously
- **Leaderboard & Selection**: Automated ranking helps authorities shortlist candidates based on performance

## Use Cases

- **Sports Academies**: Evaluate fitness levels of applicants (e.g., cricket academies using fitness tests similar to YoYo tests)
- **Schools & Colleges**: Conduct standardized fitness assessments for physical education programs
- **Recruitment**: Fitness-based selection for sports teams, defense, or physical training programs
- **Personal Training**: Athletes can self-assess and track fitness improvements over time

## Key Features

### For Athletes
- Register and participate in scheduled fitness assessments
- Upload or record exercise videos (push-ups, squats, sit-ups, lunges, jumping jacks, vertical jumps)
- Receive automated AI analysis with rep count, form score, and feedback
- View progress dashboard with badges, trends, and goal tracking
- Compare performance on leaderboards
- Track historical results and improvements

### For Authorities (Academies/Training Centers)
- Create and schedule fitness assessments for specific dates and times
- Define exercise types and weights for each assessment
- View participant registrations and leaderboards
- Shortlist candidates based on performance metrics
- Export results as CSV/PDF for record-keeping
- Manage multiple assessments across different sports or programs

### For Administrators
- Monitor platform-wide statistics (total athletes, authorities, sessions)
- View AI quality metrics and cheat detection rates
- Manage users and system activity
- Access analytics dashboard for platform health

## Supported Exercises

- **Push-ups**: Upper body strength and endurance
- **Squats**: Lower body strength and form
- **Sit-ups**: Core strength assessment
- **Lunges**: Balance and leg strength
- **Jumping Jacks**: Cardiovascular endurance
- **Vertical Jump**: Explosive power measurement

## Architecture Overview

```text
AI-Sports-Talent-Assessment/
  backend/
    main.py                # FastAPI app entry point
    config.py              # Environment and app config
    database.py            # MongoDB lifespan, indexes, seed data
    models.py              # Pydantic request/response models
    dependencies.py        # Auth dependencies
    auth/                  # Login, register, profile, password reset
      routes.py
      utils.py
    results/               # Results, progress, notifications, exports
      routes.py
      utils.py
    uploads_module/        # Video upload and processing jobs
      routes.py
      processor.py
    tests_module/          # Test templates, tests, registrations, leaderboard
      routes.py
      utils.py
    admin_module/          # Admin analytics and management
      routes.py
      utils.py
  frontend/
    index.html             # Landing page
    login.html             # Login with role selection
    register.html          # Registration flow
    athlete.html           # Athlete dashboard
    admin.html             # Authority/Admin dashboard
    styles/                # CSS files with design system
  pose_module/             # AI pose analysis modules
    pose_analyzer.py       # MediaPipe-based exercise analysis
  .env                     # Environment configuration
  requirements.txt         # Python dependencies
```

## Tech Stack

### Backend
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Motor**: Async MongoDB driver
- **Pydantic**: Data validation
- **MongoDB**: NoSQL database for users, sessions, results, tests

### AI/Computer Vision
- **MediaPipe**: Pose landmark detection
- **OpenCV**: Video processing
- **NumPy**: Numerical computations

### Frontend
- **HTML/CSS/JavaScript**: Static frontend served by FastAPI
- **Chart.js**: Data visualization for dashboards
- **Vanilla JS**: No framework dependencies for simplicity

## Prerequisites

- Python 3.10+
- MongoDB (local or Atlas)
- Modern web browser

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Dnaveed/AI-Sports-Talent-Assessment.git
cd AI-Sports-Talent-Assessment
```

### 2. Install Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=AthleteAI
MONGO_DB=athleteai
SECRET_KEY=your-secret-key-change-in-production
SCORING_MODE=hybrid

# Optional: SMTP for password reset emails
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# Frontend URL for password reset links
FRONTEND_URL=http://localhost:3000
```

### 4. Start MongoDB

Ensure MongoDB is running locally or use MongoDB Atlas cloud database.

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload
```

Backend will run at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 6. Start the frontend

Open `frontend/index.html` in a browser or use a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Frontend will run at: http://localhost:3000

## Default Admin Account

The system automatically creates an admin account on first run:
- **Email**: admin@athleteai.com
- **Password**: admin123

Athletes and authorities can register through the registration page.

## How It Works

### 1. Authority Creates Assessment
- Authority logs in and creates a fitness test
- Defines exercises (e.g., push-ups, squats) with weights
- Sets date, time, duration, and participant limits
- Publishes the assessment

### 2. Athletes Register
- Athletes browse available assessments
- Register for tests they want to participate in
- Receive notifications about upcoming tests

### 3. Athletes Take Test
- Athletes upload or record exercise videos
- AI analyzes videos using MediaPipe pose estimation
- System counts reps, evaluates form, detects cheating
- Results are stored with detailed metrics

### 4. Leaderboard & Selection
- Automated leaderboard ranks participants by performance
- Authority reviews results and shortlists candidates
- Export results as CSV/PDF for records

### 5. Progress Tracking
- Athletes view historical performance
- Track improvements with badges and trends
- Set personal goals and monitor progress

## AI Analysis Pipeline

1. **Video Upload**: Athlete uploads exercise video
2. **Pose Detection**: MediaPipe extracts 33 body landmarks per frame
3. **Exercise Recognition**: System identifies exercise type
4. **Rep Counting**: Tracks movement phases (up/down) to count reps
5. **Form Scoring**: Evaluates joint angles, alignment, and range of motion
6. **Quality Assessment**: Detects partial reps, improper form, suspicious patterns
7. **Cheat Detection**: Flags invalid attempts based on movement analysis
8. **Feedback Generation**: Provides actionable improvement suggestions
9. **Performance Metrics**: Calculates fitness level, percentile, grade

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user
- `PUT /api/auth/profile` - Update profile and goals
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password with token

### Results
- `GET /api/results` - Get user's results
- `GET /api/results/{id}` - Get specific result
- `GET /api/progress` - Get progress summary with goals
- `GET /api/notifications` - Get notifications
- `GET /api/results/export` - Export results as CSV/PDF

### Uploads
- `POST /api/sessions/upload` - Upload video for analysis
- `GET /api/jobs/{id}` - Check processing status

### Tests
- `GET /api/tests` - List assessments
- `POST /api/tests` - Create assessment (authority only)
- `POST /api/tests/{id}/register` - Register for assessment
- `GET /api/tests/{id}/leaderboard` - View leaderboard
- `GET /api/tests/{id}/leaderboard/export` - Export leaderboard

### Admin
- `GET /api/admin/stats` - Platform statistics
- `GET /api/admin/athletes` - List all athletes
- `GET /api/admin/authorities` - List all authorities

## Troubleshooting

### MongoDB Connection Issues
- Verify `MONGO_URI` in `.env` is correct
- For Atlas, ensure IP whitelist includes your IP
- Check network connectivity

### Video Processing Fails
- Ensure `mediapipe` is installed: `pip install mediapipe`
- Check video format (MP4, MOV, WebM, AVI supported)
- Verify video shows full body with good lighting

### Frontend API Calls Fail
- Confirm backend is running on port 8000
- Check browser console for CORS errors
- Verify API URL in frontend JavaScript

### Password Reset Not Working
- Configure SMTP settings in `.env`
- Use app-specific password for Gmail
- Check spam folder for reset emails

## Project Structure

- **Modular Backend**: Separate modules for auth, results, uploads, tests, admin
- **MongoDB Collections**: users, test_sessions, processing_jobs, analysis_results, tests, test_registrations, password_resets
- **Role-Based Access**: Athlete, Authority, Admin with different permissions
- **Async Processing**: Background video processing with job status tracking

## Future Enhancements

- Real-time voice feedback during recording
- Injury risk detection and warnings
- Mobile app (React Native/Flutter)
- Wearable device integration
- AI-generated personalized training plans
- Social features and challenges
- Multi-language support
- Advanced analytics with ML predictions

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue on GitHub.

---

**AthleteAI** - Transforming fitness assessment through AI-powered computer vision.
