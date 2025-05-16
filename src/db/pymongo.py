from pymongo import MongoClient

class MongoConn:
    """Responsible for loading data into target systems"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
    
    def connect_to_mongodb(self):
        """Establish connection to MongoDB"""
        try:
            uri = f"mongodb://{self.config['mongo']['host']}:{self.config['mongo']['port']}"
            
            connection_options = {
                'socketTimeoutMS': 3600000,
                'connectTimeoutMS': 30000,
                'serverSelectionTimeoutMS': 30000
            }
            
            self.client = MongoClient(
                uri,
                username=self.config['mongo']['username'],
                password=self.config['mongo']['password'],
                authSource=self.config['mongo']['database'],
                **connection_options
            )
            
            self.client.admin.command('ping')
            self.db = self.client[self.config['mongo']['database']]
            self.collection = self.db[self.config['mongo']['collection']]
            
            return True
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return False
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()