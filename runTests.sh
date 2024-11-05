#!/bin/zsh

# Execute the executable
g++ PRNGTester.cpp -o r
./r

# Run the Python script
python3 StatsDoer.py
rm r
