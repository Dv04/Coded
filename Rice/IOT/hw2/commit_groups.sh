#!/bin/bash
cd /Users/apple/Coded

echo "Committing .gitignore..."
git add .gitignore
git commit -m "chore: update gitignore for data and checkpoints"

echo "Committing .DS_Store files..."
find . -name ".DS_Store" | xargs git add
git commit -m "chore: update .DS_Store files across directories"

echo "Committing IOT hw0 changes..."
git add Rice/IOT/hw0* Rice/IOT/hw0/
git commit -m "refactor(iot): move hw0 files into dedicated directory"

echo "Committing IOT hw2 changes..."
git add Rice/IOT/hw2/
git commit -m "feat(iot): add hw2 implementation files"

echo "Committing DLVL changes..."
git add Rice/DLVL/
git commit -m "feat(dlvl): initialize DLVL course directories and hw1/hw2"

echo "Committing ML_G changes..."
git add Rice/ML_G/Project/ Rice/ML_G/hw2/
git commit -m "feat(ml-g): add Project and hw2 directories"

echo "Committing NLP HW1 changes..."
git add Rice/NLP/HW1/
git commit -m "feat(nlp): update HW1 notebook and add submission zip"

echo "Committing PC HW1 changes..."
git add Rice/PC/HW1/
git commit -m "refactor(pc): extract quicksort2 assignment from zip"

echo "Done formatting main groups."
git status -s
