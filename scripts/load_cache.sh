#!/bin/bash

if [ ! -d "local_cache/" ]; then
    tar -xzf local_cache.tar.gz
    echo "Extraction completed successfully."
else
    echo "Directory local_cache/ already exists. Skipping extraction. Manually delete local_cache/ if you really want to reset your cache."
fi
