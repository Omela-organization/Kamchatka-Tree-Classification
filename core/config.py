import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    SENTRY_SDK = os.environ.get('SENTRY_SDK')
