main_operation=$1
main_function=$2
main_data=$3
main_category=$4
main_category_num=$5
main_dict_num=$6
main_dict_thre=$7


train_file_prefix=../${main_data}/${main_category}.train.
test_file_prefix=../${main_data}/${main_category}.test.
#<<BLOCK
python filter_style_ngrams.py $train_file_prefix $main_category_num $main_function
<<BLOCK
for((i=0;i < $main_category_num; i++))
do
	python preprocess_train.py ${train_file_prefix}${i} ${train_file_prefix}${i} ${main_function} ${main_dict_num} ${main_dict_thre}
	sh build_data.sh ${train_file_prefix}${i}.data.${main_function}	
done
#BLOCK
train_data_file=../${main_data}/train.data.${main_function}
test_data_file=../${main_data}/test.data.${main_function}
dict_train_file=../${main_data}/zhi.dict.${main_function}
mv $train_data_file ${train_data_file}.old
mv $test_data_file ${test_data_file}.old

cat ${train_file_prefix}*.data.${main_function}.train >> $train_data_file
cat ${train_file_prefix}*.data.${main_function}.test >> $test_data_file

python shuffle.py $train_data_file
python shuffle.py $test_data_file
cat ${test_data_file}.shuffle >>${train_data_file}.shuffle
python create_dict.py ${train_data_file} $dict_train_file
BLOCK


