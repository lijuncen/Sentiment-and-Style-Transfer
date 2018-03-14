main_operation=$1
main_function=$2
main_data=$3
main_category=$4
main_category_num=$5
main_dict_num=$6
main_dict_thre=$7


train_file_prefix=../${main_data}/${main_category}.train.
test_file_prefix=../${main_data}/${main_category}.test.
for((i=0;i<$main_category_num;i++))
do
	python preprocess_test.py ${test_file_prefix}${i} ${train_file_prefix}${i} $main_function $main_dict_num $main_dict_thre
	python filter_template_test.py ${test_file_prefix}${i} ${main_function}
	python filter_template.py ${train_file_prefix}${i} ${main_function}
done
<<BLOCK
python prepare_test_data1.py sentiment.test.0 sentiment.train.0
python prepare_test_data1.py sentiment.test.1 sentiment.train.1
#cp sentiment.train.1.data.test sentiment.test.1.data
#cp sentiment.train.0.data.test sentiment.test.0.data
python filter_template_test.py sentiment.test.1
python filter_template_test.py sentiment.test.0
python filter_template.py sentiment.train.1
python filter_template.py sentiment.train.0
#cp *.template /data1/qspace/juncenli/template_style_transform/data/style_transfer/noise_slot_data/
BLOCK
