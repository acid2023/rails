#!/bin/bash

# Function to start the Rails code container
start_rails_code() {
  docker exec -it rails_code bash
}

# Function to start the PostgreSQL container
start_postgre_db() {
  docker exec -it postgre_db bash
}

# Function to print the usage instructions
print_usage() {
  echo "Usage: ./start_containers.sh [option]"
  echo "Options:"
  echo "  -r, --rails    Start the Rails code container"
  echo "  -p, --postgres Start the PostgreSQL container"
  echo "  -h, --help     Display this help message"
}

# Check if the containers are already started
is_rails_code_started() {
  docker ps -q --filter "name=rails_code" | grep -q .
}

is_postgres_db_started() {
  docker ps -q --filter "name=postgre_db" | grep -q .
}

# Parse the command-line arguments
case "$1" in
  -r | --rails)
    if is_rails_code_started; then
      docker exec -it rails_code bash
    else
      docker-compose up -d
      start_rails_code
      docker-compose down
    fi
    ;;
  -p | --postgres)
    if is_postgres_db_started; then
      docker exec -it postgre_db bash
    else
      docker-compose up -d
      start_postgre_db
      docker-compose down
      
    fi
    ;;
  -h | --help)
    print_usage
    ;;
  *)
    print_usage
    ;;
esac