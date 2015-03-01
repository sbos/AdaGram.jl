if [[ `uname` == 'Darwin' ]]; then
	DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(dirname $0)/lib julia "$@"
elif [[ `uname` == 'Linux' ]]; then
	LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(dirname $0)/lib julia "$@"
fi
