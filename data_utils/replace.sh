for folder in *mozilla*; do
	cd $folder
	for f in *.wav.*; do 
	    mv -- "$f" "${f/.wav*/}.lab"
	done
	cd ..
done
