#!/bin/bash

pushd .
cd ../data/
sort dataset.txt | uniq >dataset.t
rm dataset.txt
mv dataset.t dataset.txt
popd

