#!/bin/bash
set -e

echo -e "\n==> Waiting for MongoDB to be ready...\n"
until mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; do
  echo "==> MongoDB is not ready yet... waiting 1 second"
  sleep 1
done

echo -e "\n==> MongoDB is ready! Starting database initialization...\n"

DB_NAME=${MONGO_DB}
DB_USER=${MONGO_USERNAME}
DB_PASS=${MONGO_PASSWORD}

echo -e "\n==> Creating database: $DB_NAME\n"
echo -e "\n==> With username: $DB_USER\n"

mongosh admin --quiet --eval "
  db = db.getSiblingDB('$DB_NAME');
  
  if (!db.getUser('$DB_USER')) {
    db.createUser({
      user: '$DB_USER',
      pwd: '$DB_PASS',
      roles: [{ role: 'dbOwner', db: '$DB_NAME' }]
    });
    print('==> User $DB_USER successfully created in database $DB_NAME');
  } else {
    print('==> User $DB_USER already exists');
  }
  
  db.createCollection('system_info');
  db.system_info.insertOne({
    initialization_completed: true,
    timestamp: new Date(),
    version: '1.0'
  });
  
  print('==> Initial collection created!');
  print('==> Database setup completed successfully!');
"

echo -e "\n==> MongoDB initialization completed!\n"