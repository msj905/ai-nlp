"""
-------------------------------提交代代码
git status
git add .
git commit -m ''

git push

--------------------------------解决冲突

git pull

解决冲突  处理冲突代码

git commit -m ''


------------------------------查看log

git log

git reflog

git log --pretty=oneline

--------------------------------tag操作

git tag

gti tag -a v1 -m 'v1'

git show v1


git push origin v1

git push origin --tags

-----------------------------回退版本

git reset --head HEAD^

git reset --head commit_id

git reset --head HEAD~100

-------------------------------创建分支
git branch -a

git branch 1

git checkout branch1
git push --set-upstrm origin branch1
------------------------合并分支
git checkout master
git merge branch1
-------------------------------------删除分支
git branch -d brnach1
git push origin :branch1






"""