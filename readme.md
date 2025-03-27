# MetBot - AI-Powered Insurance Claims Assistant

MetBot is an intelligent chatbot designed to streamline the car insurance claims process. It uses AI to guide users
through filing claims, collecting necessary information, and detecting potential fraud.

## Features

- **Conversational Interface**: Natural language processing for a seamless user experience
- **Document Upload**: Support for uploading photos and documents related to claims
- **Voice Input**: Speech-to-text functionality for hands-free interaction
- **Fraud Detection**: AI-powered analysis to identify potentially fraudulent claims
- **Claim Summary**: Comprehensive summary of all collected information
- **Sample User Data**: Pre-populated information for demo purposes

## Technology Stack

- Django (Backend)
- Django REST Framework (API)
- Google Gemini AI (Natural Language Processing)
- JavaScript/jQuery (Frontend)
- PostgreSQL (Database)
- Bootstrap (UI Framework)

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- Google Gemini API key

### Installation

1. Clone the repository
    ```bash
    git clone https://github.com/yourusername/metbot.git
    cd metbot
    ```

2. Create and activate a virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file in the project root with the following variables:
    ```bash
    SECRET_KEY=your_secret_key
    DEBUG=True
    ALLOWED_HOSTS=["localhost", "127.0.0.1"]
    INSTALLED_APPS=["django.contrib.admin", "django.contrib.auth", "django.contrib.contenttypes", "django.contrib.sessions", "django.contrib.messages", "django.contrib.staticfiles", "rest_framework", "claims_bot"]
    MIDDLEWARE=["django.middleware.security.SecurityMiddleware", "django.contrib.sessions.middleware.SessionMiddleware", "django.middleware.common.CommonMiddleware", "django.middleware.csrf.CsrfViewMiddleware", "django.contrib.auth.middleware.AuthenticationMiddleware", "django.contrib.messages.middleware.MessageMiddleware", "django.middleware.clickjacking.XFrameOptionsMiddleware"]
    DATABASE_ENGINE=django.db.backends.postgresql
    DATABASE_NAME=metbot
    DATABASE_USER=your_db_user
    DATABASE_PASS=your_db_password
    DATABASE_HOST=localhost
    DATABASE_PORT=5432
    SESSION_COOKIE_AGE=86400
    GOOGLE_API_KEY=your_gemini_api_key
   ```

5. Set up the database
   ```bash
   python manage.py migrate
   ```

6. Run the development server
   ```bash
   python manage.py runserver
   ```

7. Access the application at http://localhost:8000

## Usage

1. Start a new claim by clicking "Start New Claim"
2. Enter your name or select a sample user
3. Describe the incident that occurred
4. Answer the chatbot's questions about the claim
5. Upload any relevant documents when prompted
6. Review the claim summary when all information is collected

## Sample Users

For demo purposes, the application includes pre-populated information for:

- Deepak (Toyota Camry)
- Rajiv (Honda Civic)
- Piyush (Ford Mustang)

## License

This project is licensed under the MIT License - see the LICENSE file for details.