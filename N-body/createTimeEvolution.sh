#!/bin/sh

rootName="grav"
let fileLength=$( cat grav.txt | wc -l )
#let i=1
#while((i<=fileLength))
for i in $(seq -f "%06g" 1 $( cat grav.txt | wc -l ) )
do
	fileName="${rootName}.${i}.csv"
#	echo -en "Writing into ${fileName}... "

	buffer="$( head --lines=$i grav.txt | tail --lines=1 | cut -f 1 --complement  | sed 's/\t/, /g')"	
	numberOfParticles=$[ $( echo ${buffer} | wc -w ) / 3 ]
	
#	echo "[number of particles = ${numberOfParticles}]"
	
	echo "x,y,z" > ${fileName}
	for j in $(seq 1 ${numberOfParticles} )
	do
		echo ${buffer} | sed 's/, /,/g' | cut -f $[3*$j-2],$[3*$j-1],$[3*$j] -d, >> ${fileName}
	done
#	echo -en "done!\n"

done
