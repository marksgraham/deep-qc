#!/bin/bash
curl -L -o data.zip https://www.dropbox.com/sh/ffmzwqegqllzlmo/AABT2JCuaH_cL4f1_FjATH54a?dl=1
curl -L -o models.zip  https://www.dropbox.com/sh/h14fryq7gzs4ecl/AADUAYKJkhkQlYbslBLfjemza?dl=1

unzip -nx -d data/ data.zip -x /
mkdir code/models/
unzip -nx -d code/models/ models.zip -x /

rm data.zip
rm models.zip