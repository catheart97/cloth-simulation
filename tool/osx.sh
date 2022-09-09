#!/bin/zsh

export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX=${CC}"++"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"