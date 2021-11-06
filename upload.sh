#!/bin/sh

printf "current working directory:\n"
pwd

#security warning
printf "Are you sure you want to commit latest changes? [y/n]\n"
read OPTION
if [ "$OPTION" = "n" ] ; then
	printf "Exiting.\n"
	exit 0
fi

#enter commit message
printf "Proceed to commit.\n"
printf "Please enter commit message:\n"
read COMMIT_MESSAGE

git add .
git commit -m "$COMMIT_MESSAGE"
git push origin master

#exiting message
printf "Commit finished."