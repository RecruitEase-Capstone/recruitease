from dotenv import load_dotenv
import os

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    return {
        'mongo': {
            'host': os.getenv('MONGO_HOST'),
            'port': os.getenv('MONGO_PORT'),
            'database': str(os.getenv('MONGO_DB')),
            'collection': os.getenv('MONGO_COLLECTION'),
            'username': os.getenv('MONGO_USERNAME'),
            'password': os.getenv('MONGO_PASSWORD')
        }
    }