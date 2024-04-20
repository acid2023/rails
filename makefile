.PHONY: import-db-updates export-db-dump import-routes-pkl export-routes-pkl copy-models-from-local copy-models-to-local

import-db-updates:
    docker exec -i postgres_db psql -U postgres -d trains -f /path/to/updates.sql

export-db-dump:
    docker exec -t postgres_db pg_dump -U postgres -d trains > /opt/homebrew/var/postgresql@14/postgre_docker/database_dump.sql

import-routes-pkl:
    docker cp /Users/sergeykuzmin/projects/rails/rails/pkl_files/routes.pkl rails_code:/app/pkl_files/routes.pkl

export-routes-pkl:
    docker cp rails_code:/app/pkl_files/routes.pkl /Users/sergeykuzmin/projects/rails/rails/pkl_files/routes.pkl

# Copy models from local folder to container
copy-models-from-local:
    docker cp /Users/sergeykuzmin/projects/rails/rails/m_learning/models rails_code:/app/rails/m_learning

# Copy models from container to local folder
copy-models-to-local:
    docker cp rails_code:/app/rails/m_learning/models /Users/sergeykuzmin/projects/rails/rails/m_learning