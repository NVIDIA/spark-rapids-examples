# Follow these steps to package the Python zip file
rm -fr samples.zip
cd agaricus/python ; zip -r ../../samples.zip com ; cd ../..
cd mortgage/python ; zip -r ../../samples.zip com ; cd ../..
cd taxi/python ; zip -r ../../samples.zip com ; cd ../..
cd utility/python ; zip -r ../../samples.zip com ; cd ../..
