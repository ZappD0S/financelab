#!/bin/bash

gcc -I/usr/include/postgresql/server -fpic -O2 -flto -c get_next_trading_ts.c
gcc -shared get_next_trading_ts.o -o get_next_trading_ts.so
sudo cp get_next_trading_ts.so /usr/lib/postgresql/