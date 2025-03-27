import json
import os
from pathlib import Path

import environ

ENV = environ.Env()
BASE_DIR = Path(__file__).resolve().parent.parent
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

SECRET_KEY = ENV('SECRET_KEY')
DEBUG = ENV.bool('DEBUG')

ALLOWED_HOSTS = json.loads(ENV('ALLOWED_HOSTS'))

INSTALLED_APPS = json.loads(ENV('INSTALLED_APPS'))

MIDDLEWARE = json.loads(ENV('MIDDLEWARE'))

ROOT_URLCONF = 'metbot.urls'

LOGIN_URL = '/admin/login/'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'metbot.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': ENV('DATABASE_ENGINE'),
        'NAME': ENV('DATABASE_NAME'),
        'USER': ENV('DATABASE_USER'),
        'PASSWORD': ENV('DATABASE_PASS'),
        'HOST': ENV('DATABASE_HOST'),
        'PORT': ENV('DATABASE_PORT'),
    }
}
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Kolkata'

USE_I18N = True

USE_TZ = True

STATIC_ROOT = os.path.join(BASE_DIR, 'static/')
STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

SESSION_COOKIE_AGE = int(ENV('SESSION_COOKIE_AGE'))

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
