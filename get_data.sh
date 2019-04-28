wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nJDGa-chNuYPiwoJKJm4ZsBzhIWpX4P_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nJDGa-chNuYPiwoJKJm4ZsBzhIWpX4P_" -O celeba_resized.zip && rm -rf /tmp/cookies.txt 


# warning: takes some time...
rm -r celeba_resized
unzip -q celeba_resized.zip
