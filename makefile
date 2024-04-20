.PHONY: import-db-updates export-db-dump import-routes-pkl export-routes-pkl copy-models-from-local copy-models-to-local make-dump

import-db-updates:
	docker exec -i postgre_db psql -U postgres -d trains -f /docker-entrypoint-initdb.d/database_dump.sql
	echo "db dump from local folders imported to container"

export-db-dump:
	docker exec -t postgre_db pg_dump -U postgres -d trains > /opt/homebrew/var/postgresql@14/postgre_docker/database_dump.sql
	echo "db dump from container exported to local folder"

import-routes-pkl:
	docker cp /Users/sergeykuzmin/projects/rails/rails/pkl_files/routes.pkl rails_code:/app/rails/pkl_files/routes.pkl
	echo "pkl file with routes data imported from local folder to container"

export-routes-pkl:
	docker cp rails_code:/app/rails/pkl_files/routes.pkl /Users/sergeykuzmin/projects/rails/rails/pkl_files/routes.pkl
	echo "pkl file with routes data exported from container to local folder"

# Copy models from local folder to container
copy-models-from-local:
	docker cp /Users/sergeykuzmin/projects/rails/rails/m_learning/models rails_code:/app/rails/m_learning
	echo "models copied from local folder to container"

# Copy models from container to local folder
copy-models-to-local:
	docker cp rails_code:/app/rails/m_learning/models /Users/sergeykuzmin/projects/rails/rails/m_learning
	echo "models copied from container to local folder"

make-dump:
	pg_dump -U postgres -h localhost -f /opt/homebrew/var/postgresql@14/postgre_docker/database_dump.sql trains --no-owner
	echo "Dump created successfully"