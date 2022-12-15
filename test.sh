#!/bin/bash
build(){
	if [ -d build ];then
		rm -rf build
		#echo "build dir is exist, rm it"
	fi
	mkdir build
	cd build
	cmake .. -DOPType=$1
	make 
}
run(){
	./main
}

build_and_run(){
	build $1
	run
	return $?
}

#main function
run_failed=0
result_failed=0
successed=0
total_case=0
pwd=$PWD
cd $pwd/operator
res=$(find ./ -name "*.h")
total_case=$(ls $res | wc -l)
for name in ${res[@]}
do
	temp=${name##*/}
	real_name=${temp%.*}
	echo $real_name
	cd $pwd
	build_and_run $real_name
	ret=$?
	if [ $ret -eq 0 ];then
	    successed_list[$successed]=$real_name
		successed=`expr $successed + 1`
	elif [ $ret -eq 1 ];then
		result_failed_list[$result_failed]=$real_name
		result_failed=`expr $result_failed + 1`
	else
		run_failed_list[$run_failed]=$real_name
		run_failed=`expr $run_failed + 1`

	fi
done

echo "total_case:$total_case, successed:$successed, result_failed:$result_failed, run_failed:$run_failed"
echo "successed list:${successed_list[@]}"
echo "result_failed:${result_failed_list[@]}"
echo "run_failed:${run_failed_list[@]}"
