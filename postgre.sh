#!/bin/bash

if pgrep "postgres" > /dev/null
then
    echo "PostgreSQL is running. Stopping..."
    brew services stop postgresql
else
    echo "PostgreSQL is not running. Starting..."
    brew services start postgresql
fi