---
date: 2024-06-03
tags:
  - obsidian
dir: basic
share: "true"
---

# 基础

|      | 语法                                           |
| ---- | -------------------------------------------- |
| 标题   | `#` + `space`                                |
| 段落   | 空白行                                          |
| 换行   | `space*2`+`enter`                            |
| 加粗   | **Bold** or __Bold__                         |
| 引用   | `>`                                          |
| 列表   | `number + .`                                 |
| 代码   | `code`                                       |
| 分隔线  | *** or --- or ___                            |
| 链接   | [url](https://markdown.com.cn/basic-syntax/) |
| 图片   | ![图片]（）                                      |
| 转义字符 | \                                            |

# 扩展

### 代码块

~~~C
#include<stdio.h>
int main()
{
	printf("Hello World!");
	return 0;
}
~~~
### 表格
| 语法    | 描述    |
| :-----: | :----: |
| Hello | World |

### 脚注
这是一个示例[^1]
([^1]:this is a example)

### 标题编号 
`### Heading {#custom-id}`

### 删除线

~~删除线~~ 

### 任务列表
- [x] 学习MarkDown
- [x] 背单词
