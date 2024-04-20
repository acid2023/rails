#!/bin/bash

pg_dump -U postgres -h localhost  -f /opt/homebrew/var/postgresql@14/postgre_docker/database_dump.sql trains --no-owner
echo "Dump created successfully:"