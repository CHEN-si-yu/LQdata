#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

# ------------ check prams ------------
if [ $# -ne 2 ]; then
    echo -e "${RED}参数错误${NC}"
    echo -e "用法: bash $0 <本地代码目录> <GitLab远程仓库地址>"
    echo -e "示例: bash $0 /home/user2/code git@192.168.253.106:gitpublic/datamake128-user2.git"
    exit 1
fi

WORK_DIR="$1"
GIT_REMOTE="$2"
DATE_TAG=$(date "+%Y%m%d-%H%M")
RELEASE_BRANCH="release-$DATE_TAG"
COMMIT_MESSAGE="Initial release commit: auto-pushed on $(date '+%Y-%m-%d %H:%M:%S')"

# get gitlab url
GITLAB_WEB_URL=$(echo "$GIT_REMOTE" | sed -E 's#git@([^:]+):#http://\1/#; s/\.git$//')

echo -e "${BLUE}准备推送目录:${NC} $WORK_DIR"
echo -e "${BLUE}远程仓库:${NC} $GIT_REMOTE [${RELEASE_BRANCH}]"

# check dir
if [ ! -d "$WORK_DIR" ]; then
    echo -e "${RED}错误: 目录不存在:${NC} $WORK_DIR"
    exit 2
fi

cd "$WORK_DIR" || { echo -e "${RED}无法进入目录:${NC} $WORK_DIR"; exit 3; }

# git init
if [ ! -d .git ]; then
    git init
    echo -e "${GREEN}已初始化 Git 仓库${NC}"
fi

# add remote origin
if ! git remote | grep -q "^origin$"; then
    git remote add origin "$GIT_REMOTE"
    echo -e "${GREEN}已添加远程仓库: origin ->${NC} $GIT_REMOTE"
fi

# check remote main branch
if git ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
    echo -e "${GREEN}远程已有 main 分支${NC}"
else
    echo -e "${YELLOW}远程不存在 main 分支，尝试创建一个空的 main 分支${NC}"
    git checkout -B main
    git commit --allow-empty -m "Initialize main branch"
    git push -u origin main
    echo -e "${GREEN}main 分支已创建并推送${NC}"
fi

# create and checkout release branch
git checkout -B "$RELEASE_BRANCH"
git add .
git commit -m "$COMMIT_MESSAGE"
git push -u origin "$RELEASE_BRANCH"
echo -e "${GREEN}已推送分支 $RELEASE_BRANCH 到远程仓库${NC}"


echo ""
echo -e "${BLUE}下一步：请登录 GitLab 创建合并请求 (Merge Request)${NC}"
echo -e "${BLUE}链接：${NC}$GITLAB_WEB_URL/-/merge_requests/new?merge_request[source_branch]=$RELEASE_BRANCH&merge_request[target_branch]=main"


