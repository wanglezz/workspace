---
title: 本地库上传Github
date: 2024-06-05
tags:
  - Git
  - 技术
dir: tech
share: "true"
author: tomato
slug: local to remote
---

## 创建远程库
在github.com创建远程库

## 创建本地库
1. 先初始化本地库
~~~bash
git init
~~~
2. 提交当前内容到工作区
~~~bash
git add .
~~~
3. 注册自己的邮箱和用户名
~~~bash
git config --global user.email "example@example.com"
git config --global user.name "yourname"
~~~
4. 添加远程库
~~~bash
git remote add origin git@github.com:yourname/project.git
~~~
5. 提交到本地仓库
~~~bash
git commit -m "note"
~~~

## ssh配置
初次提交到远程库会报错，这是因为没有配置ssh密钥并添加到ssh代理。github的ssh连接采用的是非对称加密，关于非对称加密可见[SSH](SSH.md)，配置ssh的过程如下
1. 在本地生成ssh密钥对
~~~bash
ssh-keygen -t ed25519 -C "your_email@example.com"
~~~
2. 添加私钥到ssh代理
~~~bash
eval "$(ssh-agent -s)"//启动ssh代理
ssh-add ~/.ssh/id_ed25519
~~~
3. 将公钥添加到github账户
~~~bash
cat ~/.ssh/id_ed25519.pub
~~~
在github上创建一个新的SSH key，并将生成的内容填入

## 上传远程库
完成上述步骤后应该可以成功上传远程仓库了。🎉
~~~bash
git push origin main
~~~
