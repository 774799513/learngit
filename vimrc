"去掉vi的一致性"
set nocompatible
"显示行号"
set number
" 隐藏滚动条"    
set guioptions-=r 
set guioptions-=L
set guioptions-=b
"隐藏顶部标签栏"
set showtabline=0
"设置字体"
set guifont=Monaco:h13         
syntax on   "开启语法高亮"
let g:solarized_termcolors=256  "solarized主题设置在终端下的设置"
set background=dark     "设置背景色"
colorscheme elflord 
"set nowrap  "设置不折行"
set fileformat=unix "设置以unix的格式保存文件"
set cindent     "设置C样式的缩进格式"
set tabstop=4   "设置table长度"
set shiftwidth=4        "同上"
set showmatch   "显示匹配的括号"
set scrolloff=5     "距离顶部和底部5行"
set laststatus=2    "命令行为两行"
set fenc=utf-8      "文件编码"
set backspace=2
set mouse=a     "启用鼠标"
set selection=exclusive
set selectmode=mouse,key
set matchtime=5
set ignorecase      "忽略大小写"
set incsearch
set hlsearch        "高亮搜索项"
set noexpandtab     "不允许扩展table"
set whichwrap+=<,>,h,l
set autoread
set cursorline      "突出显示当前行"

"set cursorcolumn        "突出显示当前列"
"F5自动运行
	map <F5> :call CompileRunGcc()<CR>
	func! CompileRunGcc()
        exec "w"
        if &filetype == 'c'
            exec "!g++ % -o %<"
            exec "!time ./%<"
        elseif &filetype == 'cpp'
            exec "!g++ % -o %<"
            exec "!time ./%<"
        elseif &filetype == 'java'
            exec "!javac %"
            exec "!time java %<"
        elseif &filetype == 'sh'
            :!time bash %
        elseif &filetype == 'python'
            exec "!time python3 %"
        elseif &filetype == 'html'
            exec "!firefox % &"
        elseif &filetype == 'go'
            exec "!go build %<"
            exec "!time go run %"
        elseif &filetype == 'mkd'
            exec "!~/.vim/markdown.pl % > %.html &"
            exec "!firefox %.html &"
        endif
    endfunc


autocmd BufNewFile *.py, exec ":call SetComment()" 

"注释
 func SetComment()
     call append(0,"#import tensorflow as tf") 
     call append(1,"#import numpy as np")
	 call append(2,"#import pandas as pd")
     call append(3,"#import matplotlib.pyplot as plt")
     call append(4,"#from mpl_toolkits.mplot3d import Axes3D") 
	 call append(5,"#import math,sympy,random")
	 call append(6,"#import requests,re")
	 call append(7,"#with open() as file:")
	 call append(8,"#from turtle import *")
	 call append(9,"#from bs4 import BeautifulSoup")
	 call append(10,"#with tf.Session() as sess:")
	 call append(11,"#from tensorflow.keras.models import Sequential")
	 call append(12,"#from tensorflow.keras.layers import Dense,Activation,Dropout")
	 call append(13,"#from tensorflow.optimizer import *")
	 call append(14,"#import time,datetime")  
	 call append(15,"#from pathlib import Path p=Path('.')")
	 call append(16,"#import glob,os,sys,cwd = os.getcwd()")
	 call append(17,"#F2:tree F4:notes  F5:run F8:pep8 F3:tagbar,F10:save")
 endfunc


"插件
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'Valloric/YouCompleteMe'
Plugin 'Lokaltog/vim-powerline'
Plugin 'scrooloose/nerdtree'
Plugin 'Yggdroot/indentLine'
Plugin 'jiangmiao/auto-pairs'
Plugin 'tell-k/vim-autopep8'
Plugin 'scrooloose/nerdcommenter'
Plugin 'majutsushi/tagbar'
Bundle 'altercation/vim-colors-solarized'
call vundle#end()
filetype plugin indent on 

""F2开启和关闭树"
map <F2> :NERDTreeToggle<CR>
let NERDTreeChDirMode=1
"显示书签"
let NERDTreeShowBookmarks=1
"设置忽略文件类型"
let NERDTreeIgnore=['\~$', '\.pyc$', '\.swp$']
"窗口大小"
let NERDTreeWinSize=25

"F9开启tagbar
map <F3> :TagbarToggle<CR> 
let g:tagbar_width = 25 
let g:tagbar_sort = 0
let g:tagbar_left = 1
let g:tagbar_autofocus = 1
let g:tagbar_autopreview = 1

"缩进指示线"
let g:indentLine_char='┆'
let g:indentLine_enabled = 1

"autopep8设置"
autocmd FileType python noremap <buffer> <F8> :call Autopep8()<CR>
let g:autopep8_disable_show_diff=1
"添加注释
let mapleader=','
map <F4> <leader>ci <CR>
"快速退出
map <F10> ZZ <CR>

