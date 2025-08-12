# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Loads configuration from environment variables."""
    MONGO_URI: str
    DB_NAME: str
    MODEL_PATH: str

    model_config = SettingsConfigDict(env_file=".env")

# Create a single settings instance to be used across the application
settings = Settings()
