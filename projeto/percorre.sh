#!/bin/bash -

P=$1
current=`pwd`

cd $P
for i in `find -iname '*.dcm' -type f`
do
  $current/build/projeto $i
done

cd -

# Apagar última linha?

