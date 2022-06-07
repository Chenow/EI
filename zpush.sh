
  
cd EI

git config --global user.email "ant.cheneau@hotmail.fr"
git config --global user.name "Chenow"
git add .
git commit -m "test"
git status
git push https://${GH_token}@github.com/Chenow/EI.git 