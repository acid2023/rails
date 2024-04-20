"""
Django settings for rails project.

Generated by 'django-admin startproject' using Django 4.2.3.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path

import os

# Define custom variables
RUNNING_IN_DOCKER = os.getenv('RUNNING_IN_DOCKER', 'false') == 'true'

if not RUNNING_IN_DOCKER:
    from dotenv import load_dotenv
    load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-pz$daqvkpon!xp2jk$_fp@(_ntft&$wf@5%e*33h9m9rthrh3w'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# settings.py
ALLOWED_HOSTS = ['*']  # Add the hostname or IP address of the machine


# Application definition

INSTALLED_APPS = [
    'rails',
    'channels',
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.gis',



]

if not RUNNING_IN_DOCKER:
    INSTALLED_APPS += ['django_extensions',]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'rails.urls'

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

WSGI_APPLICATION = 'rails.wsgi.application'

ASGI_APPLICATION = 'rails.asgi.application'

REDIS_CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("redis" if RUNNING_IN_DOCKER else "localhost", 6379)],
        },
    },
}

DAPHNE_CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}

CHANNEL_LAYERS = REDIS_CHANNEL_LAYERS

# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'NAME': DB_NAME,
        'HOST': 'db' if RUNNING_IN_DOCKER else 'localhost',
        'PORT': '5432',
        'USER': DB_USER,
        'PASSWORD': DB_PASSWORD,

    }
}
DATA_UPLOAD_MAX_NUMBER_FILES = 5000
GEOS_LIBRARY_PATH = '/opt/homebrew/Cellar/geos/3.12.1/lib/libgeos_c.dylib' if not RUNNING_IN_DOCKER else '/usr/lib/aarch64-linux-gnu/libgeos_c.so'
GDAL_LIBRARY_PATH = '/opt/homebrew/Cellar/gdal/3.8.4_3/lib/libgdal.34.dylib' if not RUNNING_IN_DOCKER else '/usr/lib/aarch64-linux-gnu/libgdal.so'

# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

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


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
